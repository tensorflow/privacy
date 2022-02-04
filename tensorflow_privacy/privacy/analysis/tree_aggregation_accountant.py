# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DP analysis of tree aggregation.

See Appendix D of
"Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

Functionality for computing differential privacy of tree aggregation of Gaussian
mechanism. Its public interface consists of the following methods:
  compute_rdp_tree_restart(
    noise_multiplier: float, steps_list: Union[int, Collection[int]],
    orders: Union[float, Collection[float]]) -> Union[float, Collection[float]]:
    computes RDP for DP-FTRL-TreeRestart.
  compute_rdp_single_tree(
    noise_multiplier: float, total_steps: int, max_participation: int,
    min_separation: int,
    orders: Union[float, Collection[float]]) -> Union[float, Collection[float]]:
    computes RDP for DP-FTRL-NoTreeRestart.
  compute_zcdp_single_tree(
    noise_multiplier: float, total_steps: int, max_participation: int,
    min_separation: int) -> Union[float, Collection[float]]:
    computes zCDP for DP-FTRL-NoTreeRestart.

For RDP to (epsilon, delta)-DP conversion, use the following public function
described in `rdp_accountant.py`:
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

The `noise_multiplier` is usually from `TreeCumulativeSumQuery` and
`TreeResidualSumQuery` in `dp_query.tree_aggregation_query`. The other
inputs depend on the data streaming setting (single/multi-pass) and the restart
strategy (see `restart_query`).

Example use:

(1) DP-FTRL-TreeRestart RDP:
Suppose we use Gaussian mechanism of `noise_multiplier`; a sample may appear
at most once for every epoch and tree is restarted every epoch; the number of
leaf nodes for every epoch are tracked in `steps_list`. For `target_delta`, the
estimated epsilon is:
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp_tree_restart(noise_multiplier, steps_list, orders)
  eps = rdp_accountant.get_privacy_spent(orders, rdp, target_delta)[0]

(2) DP-FTRL-NoTreeRestart RDP:
Suppose we use Gaussian mechanism of `noise_multiplier`; a sample may appear
at most `max_participation` times for a total of `total_steps` leaf nodes in a
single tree; there are at least `min_separation` leaf nodes between the two
appearance of a same sample. For `target_delta`, the estimated epsilon is:
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp_single_tree(noise_multiplier, total_steps,
    max_participation, min_separation, orders)
  eps = rdp_accountant.get_privacy_spent(orders, rdp, target_delta)[0]
"""

import functools
import math
from typing import Collection, Union

import numpy as np


def _compute_rdp_tree_restart(sigma, steps_list, alpha):
  """Computes RDP of the Tree Aggregation Protocol at order alpha."""
  if np.isinf(alpha):
    return np.inf
  tree_depths = [
      math.floor(math.log2(float(steps))) + 1
      for steps in steps_list
      if steps > 0
  ]
  return _compute_gaussian_rdp(
      alpha=alpha, sum_sensitivity_square=sum(tree_depths), sigma=sigma)


def compute_rdp_tree_restart(
    noise_multiplier: float, steps_list: Union[int, Collection[int]],
    orders: Union[float, Collection[float]]) -> Union[float, Collection[float]]:
  """Computes RDP of the Tree Aggregation Protocol for Gaussian Mechanism.

  This function implements the accounting when the tree is restarted at every
  epoch. See appendix D of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  Args:
    noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of a single
      contribution (a leaf node), which is usually set in
      `TreeCumulativeSumQuery` and `TreeResidualSumQuery` from
      `dp_query.tree_aggregation_query`.
    steps_list: A scalar or a list of non-negative intergers representing the
      number of steps per epoch (between two restarts).
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  _check_nonnegative(noise_multiplier, "noise_multiplier")
  if noise_multiplier == 0:
    return np.inf

  if not steps_list:
    raise ValueError(
        "steps_list must be a non-empty list, or a non-zero scalar, got "
        f"{steps_list}.")

  if np.isscalar(steps_list):
    steps_list = [steps_list]

  for steps in steps_list:
    if steps < 0:
      raise ValueError(f"Steps must be non-negative, got {steps_list}")

  if np.isscalar(orders):
    rdp = _compute_rdp_tree_restart(noise_multiplier, steps_list, orders)
  else:
    rdp = np.array([
        _compute_rdp_tree_restart(noise_multiplier, steps_list, alpha)
        for alpha in orders
    ])

  return rdp


def _check_nonnegative(value: Union[int, float], name: str):
  if value < 0:
    raise ValueError(f"Provided {name} must be non-negative, got {value}")


def _check_possible_tree_participation(num_participation: int,
                                       min_separation: int, start: int,
                                       end: int, steps: int) -> bool:
  """Check if participation is possible with `min_separation` in `steps`.

  This function checks if it is possible for a sample to appear
  `num_participation` in `steps`, assuming there are at least `min_separation`
  nodes between the appearance of the same sample in the streaming data (leaf
  nodes in tree aggregation). The first appearance of the sample is after
  `start` steps, and the sample won't appear in the `end` steps after the given
  `steps`.

  Args:
    num_participation: The number of times a sample will appear.
    min_separation: The minimum number of nodes between two appearance of a
      sample. If a sample appears in consecutive x, y steps in a streaming
      setting, then `min_separation=y-x-1`.
    start:  The first appearance of the sample is after `start` steps.
    end: The sample won't appear in the `end` steps after the given `steps`.
    steps: Total number of steps (leaf nodes in tree aggregation).

  Returns:
    True if a sample can appear `num_participation` with given conditions.
  """
  return start + (min_separation + 1) * num_participation <= steps + end


@functools.lru_cache(maxsize=None)
def _tree_sensitivity_square_sum(num_participation: int, min_separation: int,
                                 start: int, end: int, size: int) -> float:
  """Compute the worst-case sum of sensitivtiy square for `num_participation`.

  This is the key algorithm for DP accounting for DP-FTRL tree aggregation
  without restart, which recurrently counts the worst-case occurence of a sample
  in all the nodes in a tree. This implements a dynamic programming algorithm
  that exhausts the possible `num_participation` appearance of a sample in
  `size` leaf nodes. See Appendix D.2 (DP-FTRL-NoTreeRestart) of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  Args:
    num_participation: The number of times a sample will appear.
    min_separation: The minimum number of nodes between two appearance of a
      sample. If a sample appears in consecutive x, y size in a streaming
      setting, then `min_separation=y-x-1`.
    start:  The first appearance of the sample is after `start` steps.
    end: The sample won't appear in the `end` steps after given `size` steps.
    size: Total number of steps (leaf nodes in tree aggregation).

  Returns:
    The worst-case sum of sensitivity square for the given input.
  """
  if not _check_possible_tree_participation(num_participation, min_separation,
                                            start, end, size):
    sum_value = -np.inf
  elif num_participation == 0:
    sum_value = 0.
  elif num_participation == 1 and size == 1:
    sum_value = 1.
  else:
    size_log2 = math.log2(size)
    max_2power = math.floor(size_log2)
    if max_2power == size_log2:
      sum_value = num_participation**2
      max_2power -= 1
    else:
      sum_value = 0.
    candidate_sum = []
    # i is the `num_participation` in the right subtree
    for i in range(num_participation + 1):
      # j is the `start` in the right subtree
      for j in range(min_separation + 1):
        left_sum = _tree_sensitivity_square_sum(
            num_participation=num_participation - i,
            min_separation=min_separation,
            start=start,
            end=j,
            size=2**max_2power)
        if np.isinf(left_sum):
          candidate_sum.append(-np.inf)
          continue  # Early pruning for dynamic programming
        right_sum = _tree_sensitivity_square_sum(
            num_participation=i,
            min_separation=min_separation,
            start=j,
            end=end,
            size=size - 2**max_2power)
        candidate_sum.append(left_sum + right_sum)
    sum_value += max(candidate_sum)
  return sum_value


def _max_tree_sensitivity_square_sum(max_participation: int,
                                     min_separation: int, steps: int) -> float:
  """Compute the worst-case sum of sensitivity square in tree aggregation.

  See Appendix D.2 of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  Args:
    max_participation: The maximum number of times a sample will appear.
    min_separation: The minimum number of nodes between two appearance of a
      sample. If a sample appears in consecutive x, y steps in a streaming
      setting, then `min_separation=y-x-1`.
    steps: Total number of steps (leaf nodes in tree aggregation).

  Returns:
    The worst-case sum of sensitivity square for the given input.
  """
  num_participation = max_participation
  while not _check_possible_tree_participation(
      num_participation, min_separation, 0, min_separation, steps):
    num_participation -= 1
  candidate_sum = []
  for num_part in range(1, num_participation + 1):
    candidate_sum.append(
        _tree_sensitivity_square_sum(num_part, min_separation, 0,
                                     min_separation, steps))
  return max(candidate_sum)


def _compute_gaussian_rdp(sigma: float, sum_sensitivity_square: float,
                          alpha: float) -> float:
  """Computes RDP of Gaussian mechanism."""
  if np.isinf(alpha):
    return np.inf
  return alpha * sum_sensitivity_square / (2 * sigma**2)


def compute_rdp_single_tree(
    noise_multiplier: float, total_steps: int, max_participation: int,
    min_separation: int,
    orders: Union[float, Collection[float]]) -> Union[float, Collection[float]]:
  """Computes RDP of the Tree Aggregation Protocol for a single tree.

  The accounting assume a single tree is constructed for `total_steps` leaf
  nodes, where the same sample will appear at most `max_participation` times,
  and there are at least `min_separation` nodes between two appearance. The key
  idea is to (recurrently) count the worst-case occurence of a sample
  in all the nodes in a tree, which implements a dynamic programming algorithm
  that exhausts the possible `num_participation` appearance of a sample in
  `steps` leaf nodes.

  See Appendix D of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  Args:
    noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of a single
      contribution (a leaf node), which is usually set in
      `TreeCumulativeSumQuery` and `TreeResidualSumQuery` from
      `dp_query.tree_aggregation_query`.
    total_steps: Total number of steps (leaf nodes in tree aggregation).
    max_participation: The maximum number of times a sample can appear.
    min_separation: The minimum number of nodes between two appearance of a
      sample. If a sample appears in consecutive x, y steps in a streaming
      setting, then `min_separation=y-x-1`.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  _check_nonnegative(noise_multiplier, "noise_multiplier")
  if noise_multiplier == 0:
    return np.inf
  _check_nonnegative(total_steps, "total_steps")
  _check_nonnegative(max_participation, "max_participation")
  _check_nonnegative(min_separation, "min_separation")
  sum_sensitivity_square = _max_tree_sensitivity_square_sum(
      max_participation, min_separation, total_steps)
  if np.isscalar(orders):
    rdp = _compute_gaussian_rdp(noise_multiplier, sum_sensitivity_square,
                                orders)
  else:
    rdp = np.array([
        _compute_gaussian_rdp(noise_multiplier, sum_sensitivity_square, alpha)
        for alpha in orders
    ])
  return rdp


def _compute_gaussian_zcdp(sigma: float,
                           sum_sensitivity_square: float) -> float:
  """Computes zCDP of Gaussian mechanism."""
  return sum_sensitivity_square / (2 * sigma**2)


def compute_zcdp_single_tree(
    noise_multiplier: float, total_steps: int, max_participation: int,
    min_separation: int) -> Union[float, Collection[float]]:
  """Computes zCDP of the Tree Aggregation Protocol for a single tree.

  The accounting assume a single tree is constructed for `total_steps` leaf
  nodes, where the same sample will appear at most `max_participation` times,
  and there are at least `min_separation` nodes between two appearance. The key
  idea is to (recurrently) count the worst-case occurence of a sample
  in all the nodes in a tree, which implements a dynamic programming algorithm
  that exhausts the possible `num_participation` appearance of a sample in
  `steps` leaf nodes.

  See Appendix D of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  The Zero-Concentrated Differential Privacy (zCDP) definition is described in
  "Concentrated Differential Privacy: Simplifications, Extensions,
  and Lower Bounds" https://arxiv.org/abs/1605.02065

  Args:
    noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of a single
      contribution (a leaf node), which is usually set in
      `TreeCumulativeSumQuery` and `TreeResidualSumQuery` from
      `dp_query.tree_aggregation_query`.
    total_steps: Total number of steps (leaf nodes in tree aggregation).
    max_participation: The maximum number of times a sample can appear.
    min_separation: The minimum number of nodes between two appearance of a
      sample. If a sample appears in consecutive x, y steps in a streaming
      setting, then `min_separation=y-x-1`.

  Returns:
    The zCDP.
  """
  _check_nonnegative(noise_multiplier, "noise_multiplier")
  if noise_multiplier == 0:
    return np.inf
  _check_nonnegative(total_steps, "total_steps")
  _check_nonnegative(max_participation, "max_participation")
  _check_nonnegative(min_separation, "min_separation")
  sum_sensitivity_square = _max_tree_sensitivity_square_sum(
      max_participation, min_separation, total_steps)
  return _compute_gaussian_zcdp(noise_multiplier, sum_sensitivity_square)

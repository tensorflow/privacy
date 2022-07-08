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
"""(Deprecated) RDP analysis of the Sampled Gaussian Mechanism.

The functions in this package have been superseded by more general accounting
mechanisms in Google's `differential_privacy` package. These functions may at
some future date be removed.

Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:

  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""

import dp_accounting
import numpy as np


def _compute_rdp_from_event(orders, event, count):
  """Computes RDP from a DpEvent using RdpAccountant.

  Args:
    orders: An array (or a scalar) of RDP orders.
    event: A DpEvent to compute the RDP of.
    count: The number of self-compositions.

  Returns:
    The RDP at all orders. Can be `np.inf`.
  """
  orders_vec = np.atleast_1d(orders)

  if isinstance(event, dp_accounting.SampledWithoutReplacementDpEvent):
    neighboring_relation = dp_accounting.NeighboringRelation.REPLACE_ONE
  elif isinstance(event, dp_accounting.SingleEpochTreeAggregationDpEvent):
    neighboring_relation = dp_accounting.NeighboringRelation.REPLACE_SPECIAL
  else:
    neighboring_relation = dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE

  accountant = dp_accounting.rdp.RdpAccountant(orders_vec, neighboring_relation)
  accountant.compose(event, count)
  rdp = accountant._rdp  # pylint: disable=protected-access

  if np.isscalar(orders):
    return rdp[0]
  else:
    return rdp


def compute_rdp(q, noise_multiplier, steps, orders):
  """(Deprecated) Computes RDP of the Sampled Gaussian Mechanism.

  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.

  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  event = dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(noise_multiplier))

  return _compute_rdp_from_event(orders, event, steps)


def compute_rdp_sample_without_replacement(q, noise_multiplier, steps, orders):
  """(Deprecated) Compute RDP of Gaussian Mechanism sampling w/o replacement.

  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.

  This function applies to the following schemes:
  1. Sampling w/o replacement: Sample a uniformly random subset of size m = q*n.
  2. ``Replace one data point'' version of differential privacy, i.e., n is
     considered public information.

  Reference: Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf (A strengthened
  version applies subsampled-Gaussian mechanism)
  - Wang, Balle, Kasiviswanathan. "Subsampled Renyi Differential Privacy and
  Analytical Moments Accountant." AISTATS'2019.

  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  event = dp_accounting.SampledWithoutReplacementDpEvent(
      1, q, dp_accounting.GaussianDpEvent(noise_multiplier))

  return _compute_rdp_from_event(orders, event, steps)


def compute_heterogeneous_rdp(sampling_probabilities, noise_multipliers,
                              steps_list, orders):
  """(Deprecated) Computes RDP of Heteregoneous Sampled Gaussian Mechanisms.

  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.

  Args:
    sampling_probabilities: A list containing the sampling rates.
    noise_multipliers: A list containing the noise multipliers: the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
    steps_list: A list containing the number of steps at each
      `sampling_probability` and `noise_multiplier`.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  assert len(sampling_probabilities) == len(noise_multipliers)

  rdp = 0
  for q, noise_multiplier, steps in zip(sampling_probabilities,
                                        noise_multipliers, steps_list):
    rdp += compute_rdp(q, noise_multiplier, steps, orders)

  return rdp


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """(Deprecated) Computes delta or eps from RDP values.

  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.

  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta` must
      be `None`.

  Returns:
    A tuple of epsilon, delta, and the optimal order.

  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant._rdp = rdp  # pylint: disable=protected-access

  if target_eps is not None:
    delta, opt_order = accountant.get_delta_and_optimal_order(target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = accountant.get_epsilon_and_optimal_order(target_delta)
    return eps, target_delta, opt_order

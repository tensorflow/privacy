# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Library for computing privacy values for DP-SGD."""

import math
from typing import Optional

from absl import app
from absl import logging
import dp_accounting
from scipy import optimize


class UserLevelDPComputationError(Exception):
  """Error raised if user-level epsilon computation fails."""


def _compute_dp_sgd_user_privacy(
    num_epochs: float,
    noise_multiplier: float,
    user_delta: float,
    max_examples_per_user: int,
    used_microbatching: bool = True,
    poisson_subsampling_probability: Optional[float] = None,
) -> float:
  """Computes add-or-remove-one-user DP epsilon using group privacy.

  This privacy guarantee uses add-or-remove-one-user adjacency, and protects
  release of all model checkpoints in addition to the final model.

  Uses Vadhan (2017) "The complexity of differential privacy" Lemma 2.2.

  # TODO(b/271330804): Consider using RDP to compute group privacy.

  We use a line search to identify an example-level delta which, when the lemma
  is applied, yields the requested user-level delta, then use it to compute the
  user-level epsilon.

  Args:
    num_epochs: The number of passes over the data. May be fractional.
    noise_multiplier: The ratio of the noise to the l2 sensitivity.
    user_delta: The target user-level delta.
    max_examples_per_user: Upper bound on the number of examples per user.
    used_microbatching: If true, increases sensitivity by a factor of two.
    poisson_subsampling_probability: If not None, gives the probability that
      each record is chosen in a batch. If None, assumes no subsampling.

  Returns:
    The add-or-remove-one-user DP epsilon value using group privacy.

  Raises:
    UserLevelDPComputationError: If line search for example-level delta fails.
  """
  if num_epochs <= 0:
    raise ValueError(f'num_epochs must be positive. Found {num_epochs}.')
  if noise_multiplier < 0:
    raise ValueError(
        f'noise_multiplier must be non-negative. Found {noise_multiplier}.'
    )
  if not 0 <= user_delta <= 1:
    raise ValueError(f'user_delta must be between 0 and 1. Found {user_delta}.')
  if max_examples_per_user <= 0:
    raise ValueError(
        'max_examples_per_user must be a positive integer. Found '
        f'{max_examples_per_user}.'
    )

  if max_examples_per_user == 1:
    # Don't unnecessarily inflate epsilon if one example per user.
    return _compute_dp_sgd_example_privacy(
        num_epochs,
        noise_multiplier,
        user_delta,
        used_microbatching,
        poisson_subsampling_probability,
    )

  # The computation below to estimate user_eps works as follows.
  # We have _compute_dp_sgd_example_privacy which maps
  #   F(example_delta) -> example_eps
  # Vadhan (2017) "The complexity of differential privacy" Lemma 2.2 gives us
  #   G(example_eps, example_delta) -> user_delta
  #   H(example_eps) -> user_eps.
  # We first identify an example_delta such that
  #   G(F(example_delta), example_delta) = user_delta
  # Specifically, we use a line search in log space to solve for
  #   log(G(F(example_delta), example_delta)) - log(user_delta) = 0
  # Then we can return user_eps = H(F(example_delta)).

  log_k = math.log(max_examples_per_user)
  target_user_log_delta = math.log(user_delta)

  def user_log_delta_gap(example_log_delta):
    example_eps = _compute_dp_sgd_example_privacy(
        num_epochs,
        noise_multiplier,
        math.exp(example_log_delta),
        used_microbatching,
        poisson_subsampling_probability,
    )

    # Estimate user_eps, user_log_delta using Vadhan Lemma 2.2.
    user_eps = max_examples_per_user * example_eps
    user_log_delta = log_k + user_eps + example_log_delta
    return user_log_delta - target_user_log_delta

  # We need bounds on the example-level delta. The supplied user-level delta
  # is an upper bound. Search exponentially toward zero for lower bound.
  example_log_delta_max = target_user_log_delta
  example_log_delta_min = example_log_delta_max - math.log(10)
  user_log_delta_gap_min = user_log_delta_gap(example_log_delta_min)
  while user_log_delta_gap_min > 0:
    # Assuming that _compute_dp_sgd_example_privacy is decreasing in
    # example_delta, it is not difficult to show that if user_delta_min
    # corresponding to example_delta_min is too large, then we must reduce
    # example_delta by at least a factor of (user_delta / user_delta_min).
    # In other words, if example_log_delta_min is an upper bound, then so is
    # example_log_delta_min - user_log_delta_gap_min.
    example_log_delta_max = example_log_delta_min - user_log_delta_gap_min
    example_log_delta_min = example_log_delta_max - math.log(10)
    user_log_delta_gap_min = user_log_delta_gap(example_log_delta_min)
    if not math.isfinite(user_log_delta_gap_min):
      # User-level (epsilon, delta) DP is not achievable. This can happen
      # because as example_delta decreases, example_eps increases. So it is
      # possible for user_delta (which increases in both example_delta and
      # example_eps) to diverge to infinity as example_delta goes to zero.
      logging.warn(
          (
              'No upper bound on user-level DP epsilon can be computed with %s '
              'examples per user.'
          ),
          max_examples_per_user,
      )
      return math.inf

  # By the same logic, we can improve on the lower bound we just found, before
  # even starting the line search. We actually could do a custom line search
  # that makes use of this at each step, but brentq should be fast enough.
  example_log_delta_min -= user_log_delta_gap_min

  example_log_delta, result = optimize.brentq(
      user_log_delta_gap,
      example_log_delta_min,
      example_log_delta_max,
      full_output=True,
  )

  if not result.converged:
    raise UserLevelDPComputationError(
        'Optimization failed trying to compute user-level DP epsilon.'
    )

  # Vadhan (2017) "The complexity of differential privacy" Lemma 2.2.
  # user_delta = k * exp(k * example_eps) * example_delta
  # Given example_delta, we can solve for (k * example_eps) = user_eps.
  return max(0, target_user_log_delta - log_k - example_log_delta)


def _compute_dp_sgd_example_privacy(
    num_epochs: float,
    noise_multiplier: float,
    example_delta: float,
    used_microbatching: bool = True,
    poisson_subsampling_probability: Optional[float] = None,
) -> float:
  """Computes add-or-remove-one-example DP epsilon.

  This privacy guarantee uses add-or-remove-one-example adjacency, and protects
  release of all model checkpoints in addition to the final model.

  Args:
    num_epochs: The number of passes over the data.
    noise_multiplier: The ratio of the noise to the l2 sensitivity.
    example_delta: The target delta.
    used_microbatching: If true, increases sensitivity by a factor of two.
    poisson_subsampling_probability: If not None, gives the probability that
      each record is chosen in a batch. If None, assumes no subsampling.

  Returns:
    The epsilon value.
  """
  if num_epochs <= 0:
    raise ValueError(f'num_epochs must be positive. Found {num_epochs}.')
  if noise_multiplier < 0:
    raise ValueError(
        f'noise_multiplier must be non-negative. Found {noise_multiplier}.'
    )
  if not 0 <= example_delta <= 1:
    raise ValueError(f'delta must be between 0 and 1. Found {example_delta}.')

  if used_microbatching:
    # TODO(b/271462792)
    noise_multiplier /= 2

  event_ = dp_accounting.GaussianDpEvent(noise_multiplier)
  if poisson_subsampling_probability is not None:
    event_ = dp_accounting.PoissonSampledDpEvent(
        sampling_probability=poisson_subsampling_probability, event=event_
    )
    count = int(math.ceil(num_epochs / poisson_subsampling_probability))
  else:
    count = int(math.ceil(num_epochs))
  event_ = dp_accounting.SelfComposedDpEvent(count=count, event=event_)

  rdp_orders = (
      [1 + x / 10.0 for x in range(1, 100)]
      + list(range(11, 64))
      + [128, 256, 512, 1024]
  )
  accountant = dp_accounting.rdp.RdpAccountant(rdp_orders)  # TODO(b/271341062)
  accountant.compose(event_)
  return accountant.get_epsilon(example_delta)


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
  """Compute epsilon based on the given hyperparameters.

  This function is deprecated. It does not account for doubling of sensitivity
  with microbatching, and assumes Poisson subsampling, which is rarely used in
  practice. (See "How to DP-fy ML: A Practical Guide to Machine Learning with
  Differential Privacy", https://arxiv.org/abs/2303.00654, Sec 5.6.) Most users
  should call `compute_dp_sgd_privacy_statement` (which will be added shortly),
  which provides appropriate context for the guarantee (see the reporting
  recommendations in "How to DP-fy ML", Sec 5.3). If you need a numeric epsilon
  value under specific assumptions, it is recommended to use the `dp_accounting`
  libraries directly to compute epsilon, with the precise and correct
  assumptions of your application.

  Args:
    n: Number of examples in the training data.
    batch_size: Batch size used in training.
    noise_multiplier: Noise multiplier used in training.
    epochs: Number of epochs in training.
    delta: Value of delta for which to compute epsilon.

  Returns:
    A 2-tuple containing the value of epsilon and the optimal RDP order.
  """
  # TODO(b/265168958): Update this text for `compute_dp_sgd_privacy_statement`.
  logging.warn(
      '`compute_dp_sgd_privacy` is deprecated. It does not account '
      'for doubling of sensitivity with microbatching, and assumes Poisson '
      'subsampling, which is rarely used in practice. Please use the '
      '`dp_accounting` libraries directly to compute epsilon, using the '
      'precise and correct assumptions of your application.'
  )

  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * n / batch_size))
  accountant = dp_accounting.rdp.RdpAccountant(orders)

  event = dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(
          sampling_probability=q,
          event=dp_accounting.GaussianDpEvent(noise_multiplier),
      ),
      steps,
  )

  accountant.compose(event)
  return accountant.get_epsilon_and_optimal_order(delta)

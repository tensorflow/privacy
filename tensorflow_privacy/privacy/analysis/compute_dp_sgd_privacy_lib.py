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

import enum
import functools
import math
import textwrap
from typing import Optional

from absl import app
from absl import logging
import dp_accounting
from scipy import optimize


class UserLevelDPComputationError(Exception):
  """Error raised if user-level epsilon computation fails."""


def _logexpm1(x: float) -> float:
  """Returns log(exp(x) - 1)."""
  return x + math.log(-math.expm1(-x))


class AccountantType(enum.Enum):
  """Accountant to use for privacy accounting."""

  RDP = 'RDP'
  PLD = 'PLD'

  def get_accountant(self) -> dp_accounting.PrivacyAccountant:
    if self == AccountantType.RDP:
      return dp_accounting.rdp.RdpAccountant()
    if self == AccountantType.PLD:
      return dp_accounting.pld.PLDAccountant()
    raise ValueError(f'Unsupported Accountant type {self.value}')


def _compute_dp_sgd_user_privacy(
    num_epochs: float,
    noise_multiplier: float,
    user_delta: float,
    max_examples_per_user: int,
    used_microbatching: bool = True,
    poisson_subsampling_probability: Optional[float] = None,
    accountant_type: AccountantType = AccountantType.RDP,
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
    noise_multiplier: The ratio of the noise stddev to the l2 sensitivity.
    user_delta: The target user-level delta.
    max_examples_per_user: Upper bound on the number of examples per user.
    used_microbatching: If true, increases sensitivity by a factor of two.
    poisson_subsampling_probability: If not None, gives the probability that
      each record is chosen in a batch. If None, assumes no subsampling.
    accountant_type: The privacy accountant for computing epsilon. While this
      method supports both PLD and RDP accountants, the behavior for PLD
      accountant can sometimes be overly pessimistic. This remains to be
      investigated and fixed (b/271341062).

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
        accountant_type,
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

  target_user_log_delta = math.log(user_delta)

  # Cache example privacy values, which can be expensive.
  @functools.cache
  def get_example_eps(example_log_delta):
    return _compute_dp_sgd_example_privacy(
        num_epochs,
        noise_multiplier,
        math.exp(example_log_delta),
        used_microbatching,
        poisson_subsampling_probability,
    )

  def user_log_delta_gap(example_log_delta):
    example_eps = get_example_eps(example_log_delta)

    # Estimate user_eps, user_log_delta using Vadhan Lemma 2.2, using a tighter
    # bound seen in the penultimate line of the proof, given as
    # user_delta = (example_delta * (exp(k * example_eps) - 1)
    #               / (exp(example_eps) - 1))
    user_eps = max_examples_per_user * example_eps
    user_log_delta = (
        example_log_delta + _logexpm1(user_eps) - _logexpm1(example_eps)
    )
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
      logging.warning(
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
  return max_examples_per_user * get_example_eps(example_log_delta)


def _compute_dp_sgd_example_privacy(
    num_epochs: float,
    noise_multiplier: float,
    example_delta: float,
    used_microbatching: bool = True,
    poisson_subsampling_probability: Optional[float] = None,
    accountant_type: AccountantType = AccountantType.RDP,
) -> float:
  """Computes add-or-remove-one-example DP epsilon.

  This privacy guarantee uses add-or-remove-one-example adjacency, and protects
  release of all model checkpoints in addition to the final model.

  Args:
    num_epochs: The number of passes over the data.
    noise_multiplier: The ratio of the noise stddev to the l2 sensitivity.
    example_delta: The target delta.
    used_microbatching: If true, increases sensitivity by a factor of two.
    poisson_subsampling_probability: If not None, gives the probability that
      each record is chosen in a batch. If None, assumes no subsampling.
    accountant_type: The privacy accountant for computing epsilon.

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

  return (
      accountant_type.get_accountant()
      .compose(event_)
      .get_epsilon(example_delta)
  )


def compute_dp_sgd_privacy_statement(
    number_of_examples: int,
    batch_size: int,
    num_epochs: float,
    noise_multiplier: float,
    delta: float,
    used_microbatching: bool = True,
    max_examples_per_user: Optional[int] = None,
    accountant_type: AccountantType = AccountantType.RDP,
) -> str:
  """Produces a privacy report summarizing the DP guarantee.

  Args:
    number_of_examples: Total number of examples in the dataset. For DP-SGD, an
      "example" corresponds to one row in a minibatch. E.g., for sequence models
      this would be a sequence of maximum length.
    batch_size: The number of examples in a batch. This should be the number of
      examples in a batch, *regardless of whether/how they are grouped into
      microbatches*.
    num_epochs: The number of epochs of training. May be fractional.
    noise_multiplier: The ratio of the Gaussian noise stddev to the l2 clip norm
      at each round. It is assumed that the noise_multiplier is constant
      although the clip norm may be variable if, for example, adaptive clipping
      is used.
    delta: The target delta.
    used_microbatching: Whether microbatching was used (with microbatch size
      greater than one). Microbatching inflates sensitivity by a factor of two
      in add-or-remove-one adjacency DP. (See "How to DP-fy ML: A Practical
      Guide to Machine Learning with Differential Privacy",
      https://arxiv.org/abs/2303.00654, Sec 5.6.)
    max_examples_per_user: If the data set is constructed to cap the maximum
      number of examples each user contributes, provide this argument to also
      print a user-level DP guarantee.
    accountant_type: The privacy accountant for computing epsilon. Since the
      current approach for computing user-level privacy when using PLD
      accountant can sometimes be overly pessimistic, this method does not
      provide user-level privacy guarantee for PLD accountant_type. This remains
      to be investigated and fixed (b/271341062).

  Returns:
    A str precisely articulating the privacy guarantee.
  """

  paragraph = f"""\
DP-SGD performed over {number_of_examples} examples with {batch_size} \
examples per iteration, noise multiplier {noise_multiplier} for {num_epochs} \
epochs {'with' if used_microbatching else 'without'} microbatching"""

  if max_examples_per_user is None:
    paragraph += ', and no bound on number of examples per user.'
  else:
    paragraph += f', and at most {max_examples_per_user} examples per user.'

  paragraphs = [textwrap.fill(paragraph, width=80)]

  paragraphs.append(
      textwrap.fill(
          """\
This privacy guarantee protects the release of all model checkpoints in \
addition to the final model.""",
          width=80,
      )
  )

  paragraph = textwrap.fill(
      f"""\
Example-level DP with add-or-remove-one adjacency at delta = {delta} computed \
with {accountant_type.value} accounting:""",
      width=80,
  )

  example_eps_no_subsampling = _compute_dp_sgd_example_privacy(
      num_epochs,
      noise_multiplier,
      delta,
      used_microbatching,
      accountant_type=accountant_type,
  )
  example_eps_subsampling = _compute_dp_sgd_example_privacy(
      num_epochs,
      noise_multiplier,
      delta,
      used_microbatching,
      poisson_subsampling_probability=batch_size / number_of_examples,
      accountant_type=accountant_type,
  )

  paragraph += f"""
    Epsilon with each example occurring once per epoch:  \
{example_eps_no_subsampling:12.3f}
    Epsilon assuming Poisson sampling (*):               \
{example_eps_subsampling:12.3f}"""

  paragraphs.append(paragraph)

  inf_user_eps = False
  if max_examples_per_user is None:
    paragraphs.append(
        textwrap.fill(
            """\
No user-level privacy guarantee is possible without a bound on the number of \
examples per user.""",
            width=80,
        )
    )
  elif accountant_type == AccountantType.PLD:
    # TODO(b/271341062): Add User level DP support for PLD.
    paragraphs.append(
        textwrap.fill(
            """\
User-level DP epsilon computation is not supported for PLD accounting at this \
time. Use RDP accounting to obtain user-level DP guarantees.""",
            width=80,
        )
    )
  else:  # Case: max_examples_per_user is not None and accountant_type is RDP
    user_eps_no_subsampling = _compute_dp_sgd_user_privacy(
        num_epochs,
        noise_multiplier,
        delta,
        max_examples_per_user,
        used_microbatching,
        accountant_type=accountant_type,
    )
    user_eps_subsampling = _compute_dp_sgd_user_privacy(
        num_epochs,
        noise_multiplier,
        delta,
        max_examples_per_user,
        used_microbatching,
        poisson_subsampling_probability=batch_size / number_of_examples,
        accountant_type=accountant_type,
    )
    if math.isinf(user_eps_no_subsampling):
      user_eps_no_subsampling_str = '    inf (**)'
      inf_user_eps = True
    else:
      user_eps_no_subsampling_str = f'{user_eps_no_subsampling:12.3f}'
    if math.isinf(user_eps_subsampling):
      user_eps_subsampling_str = '    inf (**)'
      inf_user_eps = True
    else:
      user_eps_subsampling_str = f'{user_eps_subsampling:12.3f}'

    paragraph = textwrap.fill(
        f"""\
User-level DP with add-or-remove-one adjacency at delta = {delta} computed \
using {accountant_type.value} accounting and group privacy:""",
        width=80,
    )
    paragraph += f"""
    Epsilon with each example occurring once per epoch:  \
{user_eps_no_subsampling_str}
    Epsilon assuming Poisson sampling (*):               \
{user_eps_subsampling_str}"""

    paragraphs.append(paragraph)

  paragraphs.append(
      textwrap.fill(
          """\
(*) Poisson sampling is not usually done in training pipelines, but assuming \
that the data was randomly shuffled, it is believed that the actual epsilon \
should be closer to this value than the conservative assumption of an \
arbitrary data order.""",
          width=80,
      )
  )

  if inf_user_eps:
    paragraphs.append(
        textwrap.fill(
            """\
(**) A finite example-level epsilon implies a finite user-level epsilon at any \
`max_examples_per_user`, but because conversion from example-level to user-\
level DP is not exact, it is possible for the upper bound on the user-level \
epsilon to still be infinite.""",
            width=80,
        )
    )

  return '\n\n'.join(paragraphs) + '\n'


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
  """Compute epsilon based on the given hyperparameters.

  This function is deprecated. It does not account for doubling of sensitivity
  with microbatching, and assumes Poisson subsampling, which is rarely used in
  practice. (See "How to DP-fy ML: A Practical Guide to Machine Learning with
  Differential Privacy", https://arxiv.org/abs/2303.00654, Sec 5.6.) Most users
  should call `compute_dp_sgd_privacy_statement`, which provides appropriate
  context for the guarantee (see the reporting recommendations in "How to DP-fy
  ML", Sec 5.3). If you need a numeric epsilon value under specific assumptions,
  it is recommended to use the `dp_accounting` libraries directly to compute
  epsilon, with the precise and correct assumptions of your application.

  Args:
    n: Number of examples in the training data.
    batch_size: Batch size used in training.
    noise_multiplier: Noise multiplier used in training.
    epochs: Number of epochs in training.
    delta: Value of delta for which to compute epsilon.

  Returns:
    A 2-tuple containing the value of epsilon and the optimal RDP order.
  """
  logging.warning("""\
`compute_dp_sgd_privacy` is deprecated. It does not account for doubling of \
sensitivity with microbatching, and assumes Poisson subsampling, which is \
rarely used in practice. Please use `compute_dp_sgd_privacy_statement`, which \
provides appropriate context for the guarantee. To compute epsilon under \
different assumptions than those in `compute_dp_sgd_privacy_statement`, call \
the `dp_accounting` libraries directly.""")

  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = (
      [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
      + list(range(5, 64))
      + [128, 256, 512]
  )
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

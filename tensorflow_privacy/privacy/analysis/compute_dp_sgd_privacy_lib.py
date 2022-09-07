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

from absl import app
import dp_accounting


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  accountant = dp_accounting.rdp.RdpAccountant(orders)

  event = dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(q,
                                          dp_accounting.GaussianDpEvent(sigma)),
      steps)

  accountant.compose(event)

  eps, opt_order = accountant.get_epsilon_and_optimal_order(delta)

  print(
      'DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
      ' over {} steps satisfies'.format(100 * q, sigma, steps),
      end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      eps, delta))
  print('The optimal RDP order is {}.'.format(opt_order))

  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

  return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
  """Compute epsilon based on the given hyperparameters.

  Args:
    n: Number of examples in the training data.
    batch_size: Batch size used in training.
    noise_multiplier: Noise multiplier used in training.
    epochs: Number of epochs in training.
    delta: Value of delta for which to compute epsilon.

  Returns:
    Value of epsilon corresponding to input hyperparameters.
  """
  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * n / batch_size))

  return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)

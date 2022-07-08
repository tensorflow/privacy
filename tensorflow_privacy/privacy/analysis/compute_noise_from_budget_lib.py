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
from scipy import optimize


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  accountant = dp_accounting.rdp.RdpAccountant(orders)
  event = dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(q,
                                          dp_accounting.GaussianDpEvent(sigma)),
      steps)
  accountant.compose(event)
  return accountant.get_epsilon_and_optimal_order(delta)


def compute_noise(n, batch_size, target_epsilon, epochs, delta, noise_lbd):
  """Compute noise based on the given hyperparameters."""
  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * n / batch_size))

  init_noise = noise_lbd  # minimum possible noise
  init_epsilon, _ = apply_dp_sgd_analysis(q, init_noise, steps, orders, delta)

  if init_epsilon < target_epsilon:  # noise_lbd was an overestimate
    print('min_noise too large for target epsilon.')
    return 0

  cur_epsilon = init_epsilon
  max_noise, min_noise = init_noise, 0

  # doubling to find the right range
  while cur_epsilon > target_epsilon:  # until noise is large enough
    max_noise, min_noise = max_noise * 2, max_noise
    cur_epsilon, _ = apply_dp_sgd_analysis(q, max_noise, steps, orders, delta)

  def epsilon_fn(noise):  # should return 0 if guess_epsilon==target_epsilon
    guess_epsilon = apply_dp_sgd_analysis(q, noise, steps, orders, delta)[0]
    return guess_epsilon - target_epsilon

  target_noise = optimize.bisect(epsilon_fn, min_noise, max_noise)
  print(
      'DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
      ' over {} steps satisfies'.format(100 * q, target_noise, steps),
      end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      target_epsilon, delta))
  return target_noise

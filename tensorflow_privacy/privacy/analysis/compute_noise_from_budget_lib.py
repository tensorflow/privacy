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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from scipy.optimize import bisect

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)

  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  return eps, opt_order


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

  target_noise = bisect(epsilon_fn, min_noise, max_noise)
  print(
      'DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
      ' over {} steps satisfies'.format(100 * q, target_noise, steps),
      end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      target_epsilon, delta))
  return target_noise

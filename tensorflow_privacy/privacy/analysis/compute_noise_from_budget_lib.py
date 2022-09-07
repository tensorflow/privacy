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


def compute_noise(n, batch_size, target_epsilon, epochs, delta, noise_lbd):
  """Compute noise based on the given hyperparameters."""
  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * n / batch_size))

  def make_event_from_noise(sigma):
    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            q, dp_accounting.GaussianDpEvent(sigma)), steps)

  def make_accountant():
    return dp_accounting.rdp.RdpAccountant(orders)

  accountant = make_accountant()
  accountant.compose(make_event_from_noise(noise_lbd))
  init_epsilon = accountant.get_epsilon(delta)

  if init_epsilon < target_epsilon:  # noise_lbd was an overestimate
    print('noise_lbd too large for target epsilon.')
    return 0

  target_noise = dp_accounting.calibrate_dp_mechanism(
      make_accountant, make_event_from_noise, target_epsilon, delta,
      dp_accounting.LowerEndpointAndGuess(noise_lbd, noise_lbd * 2))

  print(
      'DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
      ' over {} steps satisfies'.format(100 * q, target_noise, steps),
      end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      target_epsilon, delta))
  return target_noise

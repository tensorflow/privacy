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
r"""Command-line script for computing privacy of a model trained with DP-SGD.

The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism. The mechanism's parameters are controlled by flags.

Example:
  compute_dp_sgd_privacy
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5

The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

from absl import app
from absl import flags

# Opting out of loading all sibling packages and their dependencies.
sys.skip_tf_privacy_import = True

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

FLAGS = flags.FLAGS

flags.DEFINE_integer('N', None, 'Total number of examples')
flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
flags.DEFINE_float('delta', 1e-6, 'Target delta')


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)

  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
        ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      eps, delta))
  print('The optimal RDP order is {}.'.format(opt_order))

  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

  return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
  """Compute epsilon based on the given hyperparameters."""
  q = batch_size / n  # q - the sampling ratio.
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * n / batch_size))

  return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


def main(argv):
  del argv  # argv is not used.

  assert FLAGS.N is not None, 'Flag N is missing.'
  assert FLAGS.batch_size is not None, 'Flag batch_size is missing.'
  assert FLAGS.noise_multiplier is not None, 'Flag noise_multiplier is missing.'
  assert FLAGS.epochs is not None, 'Flag epochs is missing.'
  compute_dp_sgd_privacy(FLAGS.N, FLAGS.batch_size, FLAGS.noise_multiplier,
                         FLAGS.epochs, FLAGS.delta)


if __name__ == '__main__':
  app.run(main)

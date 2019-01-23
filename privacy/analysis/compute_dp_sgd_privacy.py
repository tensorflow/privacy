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

from absl import app
from absl import flags

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent

FLAGS = flags.FLAGS

flags.DEFINE_integer('N', None, 'Total number of examples')
flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
flags.DEFINE_float('delta', 1e-6, 'Target delta')

flags.mark_flag_as_required('N')
flags.mark_flag_as_required('batch_size')
flags.mark_flag_as_required('noise_multiplier')
flags.mark_flag_as_required('epochs')


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

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


def main(argv):
  del argv  # argv is not used.

  q = FLAGS.batch_size / FLAGS.N  # q - the sampling ratio.

  if q > 1:
    raise app.UsageError('N must be larger than the batch size.')

  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + range(5, 64) +
            [128, 256, 512])

  steps = int(math.ceil(FLAGS.epochs * FLAGS.N / FLAGS.batch_size))

  apply_dp_sgd_analysis(q, FLAGS.noise_multiplier, steps, orders, FLAGS.delta)


if __name__ == '__main__':
  app.run(main)

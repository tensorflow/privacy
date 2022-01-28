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
  compute_noise_from_budget
    --N=60000 \
    --batch_size=256 \
    --epsilon=2.92 \
    --epochs=60 \
    --delta=1e-5 \
    --min_noise=1e-6

The output states that DP-SGD with these parameters should
use a noise multiplier of 1.12.
"""

from absl import app
from absl import flags

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

FLAGS = flags.FLAGS

flags.DEFINE_integer('N', None, 'Total number of examples')
flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_float('epsilon', None, 'Target epsilon for DP-SGD')
flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
flags.DEFINE_float('delta', 1e-6, 'Target delta')
flags.DEFINE_float('min_noise', 1e-5, 'Minimum noise level for search.')


def main(argv):
  del argv  # argv is not used.

  assert FLAGS.N is not None, 'Flag N is missing.'
  assert FLAGS.batch_size is not None, 'Flag batch_size is missing.'
  assert FLAGS.epsilon is not None, 'Flag epsilon is missing.'
  assert FLAGS.epochs is not None, 'Flag epochs is missing.'
  compute_noise(FLAGS.N, FLAGS.batch_size, FLAGS.epsilon, FLAGS.epochs,
                FLAGS.delta, FLAGS.min_noise)


if __name__ == '__main__':
  app.run(main)

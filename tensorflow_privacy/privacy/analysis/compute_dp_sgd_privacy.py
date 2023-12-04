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
  compute_dp_sgd_privacy \
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5 \
    --accountant_type=RDP

Prints out the privacy statement corresponding to the above parameters.
"""

from absl import app
from absl import flags
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


_NUM_EXAMPLES = flags.DEFINE_integer(
    'N', None, 'Total number of examples in the training data.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    None,
    (
        'Number of examples in a batch *regardless of how/whether they are '
        'grouped into microbatches*.'
    ),
)
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier',
    None,
    (
        'Noise multiplier for DP-SGD: ratio of Gaussian noise stddev to the '
        'l2 clip norm at each round.'
    ),
)
_NUM_EPOCHS = flags.DEFINE_float(
    'epochs', None, 'Number of epochs (may be fractional).'
)
_DELTA = flags.DEFINE_float('delta', 1e-6, 'Target delta.')
_USED_MICROBATCHING = flags.DEFINE_bool(
    'used_microbatching',
    True,
    'Whether microbatching was used (with microbatch size greater than one).',
)
_MAX_EXAMPLES_PER_USER = flags.DEFINE_integer(
    'max_examples_per_user',
    None,
    (
        'Maximum number of examples per user, if applicable. Used to compute a '
        'user-level DP guarantee.'
    ),
)
_ACCOUNTANT_TYPE = flags.DEFINE_enum(
    'accountant_type', 'RDP', ['RDP', 'PLD'], 'DP accountant to use.'
)

flags.mark_flags_as_required(['N', 'batch_size', 'noise_multiplier', 'epochs'])


def main(argv):
  del argv  # argv is not used.

  statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
      _NUM_EXAMPLES.value,
      _BATCH_SIZE.value,
      _NUM_EPOCHS.value,
      _NOISE_MULTIPLIER.value,
      _DELTA.value,
      _USED_MICROBATCHING.value,
      _MAX_EXAMPLES_PER_USER.value,
      compute_dp_sgd_privacy_lib.AccountantType(_ACCOUNTANT_TYPE.value),
  )
  print(statement)


if __name__ == '__main__':
  app.run(main)

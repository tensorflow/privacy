# Copyright 2023, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import einsum_utils


class EinsumUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      experiment_params=[
          # 1D tensors
          ([1], True),
          ([2], True),
          # 2D tensors
          ([1, 2], True),
          ([2, 1], True),
          ([2, 2], True),
          # 3D tensors
          ([2, 1, 1], True),
          ([1, 2, 1], True),
          ([1, 1, 2], True),
          ([2, 2, 1], True),
          ([2, 1, 2], True),
          ([1, 2, 2], False),
          ([2, 2, 2], False),
      ]
  )
  def test_is_batch_of_vectors(self, experiment_params):
    shape, true_result = experiment_params
    t = tf.zeros(shape)
    computed_result = einsum_utils._is_batch_of_vectors(t)
    self.assertEqual(computed_result, true_result)

  @parameterized.product(
      experiment_params=[
          (
              'ab,bc->ac',
              einsum_utils.EquationType.NO_ELLIPSES,
              ('ab', 'bc', 'ac'),
          ),
          (
              '...b,bc->...c',
              einsum_utils.EquationType.LEFT_ELLIPSES,
              ('b', 'bc', 'c'),
          ),
          (
              'ab...,bc->ac...',
              einsum_utils.EquationType.RIGHT_ELLIPSES,
              ('ab', 'bc', 'ac'),
          ),
      ]
  )
  def test_parse_einsum_equation(self, experiment_params):
    equation, true_eqn_type, true_groups = experiment_params
    (computed_eqn_type, computed_groups) = einsum_utils._parse_einsum_equation(
        equation
    )
    self.assertEqual(computed_eqn_type, true_eqn_type)
    self.assertEqual(computed_groups, true_groups)

  @parameterized.product(
      experiment_params=[
          # einsum_utils.EquationType.NO_ELLIPSES
          ('ab,bc->ac', [2, 3], None, [2, 1, 3]),
          ('adb,bc->adc', [2, 3, 4], None, [2, 3, 4]),
          ('adeb,bc->adec', [2, 3, 4, 5], None, [2, 12, 5]),
          ('abe,bec->ac', [2, 3, 4], None, [2, 1, 12]),
          ('ab,bce->ace', [2, 3], None, [2, 1, 3]),
          # einsum_utils.EquationType.LEFT_ELLIPSES
          ('...b,bc->...c', [2, 3], None, [2, 1, 3]),
          ('...b,bc->...c', [2, 3, 4], None, [2, 3, 4]),
          ('...b,bc->...c', [2, 3, 4, 5], None, [2, 12, 5]),
          ('...ab,bc->...ac', [2, 3, 4], None, [2, 3, 4]),
          ('...ab,bc->...ac', [2, 3, 4, 5], None, [2, 12, 5]),
          ('...be,bec->...c', [2, 3, 4], None, [2, 1, 12]),
          ('...b,bce->...ce', [2, 3], None, [2, 1, 3]),
          # einsum_utils.EquationType.RIGHT_ELLIPSES
          ('ab...,bc->ac...', [2, 3, 4], [0, 2, 1], [2, 4, 3]),
          ('ab...,bc->ac...', [2, 3, 4, 5], [0, 2, 3, 1], [2, 20, 3]),
          ('adb...,bc->adc...', [2, 3, 4, 5], [0, 1, 3, 2], [2, 15, 4]),
          ('adeb...,bc->adec...', [2, 3, 4, 5, 6], [0, 1, 2, 4, 3], [2, 72, 5]),
          ('abe...,bec->ac...', [2, 3, 4, 5], [0, 3, 1, 2], [2, 5, 12]),
          ('ab...,bce->ace...', [2, 3, 4], [0, 2, 1], [2, 4, 3]),
      ]
  )
  def test_reshape_einsum_inputs(self, experiment_params):
    (equation, input_shape, true_permutations, true_parsed_shape) = (
        experiment_params
    )
    num_entries = int(np.prod(input_shape))
    input_tensor = tf.reshape(tf.range(0, num_entries), input_shape)
    computed_parsed_tensor = einsum_utils._reshape_einsum_inputs(
        input_tensor,
        equation,
    )
    true_parsed_tensor = input_tensor
    if true_permutations is not None:
      true_parsed_tensor = tf.transpose(
          true_parsed_tensor, perm=true_permutations
      )
    true_parsed_tensor = tf.reshape(true_parsed_tensor, true_parsed_shape)
    self.assertAllEqual(computed_parsed_tensor, true_parsed_tensor)

  @parameterized.product(
      experiment_params=[
          # einsum_utils.EquationType.NO_ELLIPSES
          ('ab,bc->ac', [2, 3], None, [2, 1, 3]),
          ('adb,bc->adc', [2, 3, 4], None, [2, 3, 4]),
          ('adeb,bc->adec', [2, 3, 4, 5], None, [2, 12, 5]),
          ('abe,bec->ac', [2, 3, 4], None, [2, 1, 12]),
          ('ab,bce->ace', [2, 3, 4], None, [2, 1, 12]),
          # einsum_utils.EquationType.LEFT_ELLIPSES
          ('...b,bc->...c', [2, 3], None, [2, 1, 3]),
          ('...b,bc->...c', [2, 3, 4], None, [2, 3, 4]),
          ('...b,bc->...c', [2, 3, 4, 5], None, [2, 12, 5]),
          ('...ab,bc->...ac', [2, 3, 4], None, [2, 3, 4]),
          ('...ab,bc->...ac', [2, 3, 4, 5], None, [2, 12, 5]),
          ('...be,bec->...c', [2, 4], None, [2, 1, 4]),
          ('...b,bce->...ce', [2, 3, 4], None, [2, 1, 12]),
          # einsum_utils.EquationType.RIGHT_ELLIPSES
          ('ab...,bc->ac...', [2, 3, 4], [0, 2, 1], [2, 4, 3]),
          ('ab...,bc->ac...', [2, 3, 4, 5], [0, 2, 3, 1], [2, 20, 3]),
          ('adb...,bc->adc...', [2, 3, 4, 5], [0, 1, 3, 2], [2, 15, 4]),
          ('adeb...,bc->adec...', [2, 3, 4, 5, 6], [0, 1, 2, 4, 3], [2, 72, 5]),
          ('abe...,bec->ac...', [2, 3, 4], [0, 2, 1], [2, 4, 3]),
          ('ab...,bce->ace...', [2, 3, 4, 5], [0, 3, 1, 2], [2, 5, 12]),
      ]
  )
  def test_reshape_einsum_outputs(self, experiment_params):
    (equation, output_shape, true_permutations, true_parsed_shape) = (
        experiment_params
    )
    num_entries = int(np.prod(output_shape))
    output_tensor = tf.reshape(tf.range(0, num_entries), output_shape)
    computed_parsed_tensor = einsum_utils._reshape_einsum_outputs(
        output_tensor,
        equation,
    )
    true_parsed_tensor = output_tensor
    if true_permutations is not None:
      true_parsed_tensor = tf.transpose(
          true_parsed_tensor, perm=true_permutations
      )
    true_parsed_tensor = tf.reshape(true_parsed_tensor, true_parsed_shape)
    self.assertAllEqual(computed_parsed_tensor, true_parsed_tensor)


if __name__ == '__main__':
  tf.test.main()

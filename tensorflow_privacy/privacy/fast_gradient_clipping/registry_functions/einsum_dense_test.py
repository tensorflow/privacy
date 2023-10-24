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
import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import common_test_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import einsum_dense


def get_einsum_layer_generators():
  def pure_einsum_layer(equation, output_dims, bias_axes):
    return tf.keras.layers.EinsumDense(
        equation, output_dims, bias_axes=bias_axes
    )

  def sigmoid_einsum_layer(equation, output_dims, bias_axes):
    return tf.keras.layers.EinsumDense(
        equation, output_dims, bias_axes=bias_axes, activation='sigmoid'
    )

  return {
      'pure_einsum': pure_einsum_layer,
      'sigmoid_einsum': sigmoid_einsum_layer,
  }


def get_einsum_parameter_tuples():
  # (equation, input_dims, output_dims, bias_axes)
  return [
      # Case C1.
      ('ab,bc->ac', [2], [3], None),
      ('ab,bc->ac', [2], [3], 'c'),
      ('abc,cd->abd', [2, 3], [2, 4], None),
      ('abc,cd->abd', [2, 3], [2, 4], 'b'),
      ('abc,cd->abd', [2, 3], [2, 4], 'd'),
      ('abc,cd->abd', [2, 3], [2, 4], 'bd'),
      ('abc,cef->abef', [2, 3], [2, 4, 5], None),
      ('abc,cef->abef', [2, 3], [2, 4, 5], 'bf'),
      # Case C2.
      ('...b,bc->...c', [2, 3], [4], None),
      ('...b,bc->...c', [2, 3], [4], 'c'),
      ('...ab,bc->...ac', [2, 3], [2, 4], None),
      ('...ab,bc->...ac', [2, 4], [2, 4], 'c'),
      ('...abc,cd->...abd', [2, 3, 4], [2, 3, 5], None),
      ('...abc,cd->...abd', [2, 3, 4], [2, 3, 5], 'b'),
      ('...abc,cd->...abd', [2, 3, 4], [2, 3, 5], 'd'),
      ('...abc,cd->...abd', [2, 3, 4], [2, 3, 5], 'bd'),
      ('...abc,cef->...abef', [2, 3, 4], [2, 3, 5, 6], None),
      ('...abc,cef->...abef', [2, 3, 4], [2, 3, 5, 6], 'bf'),
      # Case C3.
      ('ab...,bc->ac...', [2, 3], [4, 3], None),
      ('ab...,bc->ac...', [2, 3], [4, 3], 'c'),
      ('abc...,cd->abd...', [2, 3, 4], [2, 5, 4], None),
      ('abc...,cd->abd...', [2, 3, 4], [2, 5, 4], 'b'),
      ('abc...,cd->abd...', [2, 3, 4], [2, 5, 4], 'd'),
      ('abc...,cd->abd...', [2, 3, 4], [2, 5, 4], 'bd'),
      ('abc...,cef->abef...', [2, 3, 4], [2, 5, 6, 4], None),
      ('abc...,cef->abef...', [2, 3, 4], [2, 5, 6, 4], 'bf'),
  ]


def get_einsum_layer_registry():
  einsum_registry = layer_registry.LayerRegistry()
  einsum_registry.insert(
      tfm.nlp.layers.EinsumDense,
      einsum_dense.einsum_layer_computation,
  )
  return einsum_registry


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  @parameterized.product(
      layer_name=list(get_einsum_layer_generators()),
      param_tuple=get_einsum_parameter_tuples(),
      num_microbatches=[None, 2],
      is_eager=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      layer_name,
      param_tuple,
      num_microbatches,
      is_eager,
  ):
    # Parse inputs to generate test data. Note that each batched input is a
    # reshape of a `tf.range()` call.
    equation, input_dims, output_dims, bias_axes = param_tuple
    batch_size = 4
    example_size = tf.reduce_prod(input_dims)
    example_values = tf.range(batch_size * example_size, dtype=tf.float32)
    x_batch = tf.reshape(example_values, [batch_size] + input_dims)

    # Make the layer generator via currying.
    einsum_generator = get_einsum_layer_generators()[layer_name]

    def curried_generator(a, b):
      del a, b
      return einsum_generator(equation, output_dims, bias_axes)

    # Load shared assets to all devices.
    with self.strategy.scope():
      model = common_test_utils.get_model_from_generator(
          model_generator=common_test_utils.make_one_layer_functional_model,
          layer_generator=curried_generator,
          input_dims=input_dims,
          output_dims=output_dims,
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op(x):
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=None,
          num_microbatches=num_microbatches,
          x_batch=x,
          registry=get_einsum_layer_registry(),
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, autograph=False)

    # TPUs use lower precision than CPUs, so we relax our criterion.
    # E.g., one of the TPU runs generated the following results:
    #
    #   computed_norm = 93.48296
    #   true_norm     = 93.31176
    #   abs_diff      = 0.17120361
    #   rel_diff      = 0.00183475
    #
    # which is a reasonable level of error for computing gradient norms.
    # Other trials also give an absolute (resp. relative) error of around
    # 0.05 (resp. 0.0015).
    rtol = 1e-2 if self.using_tpu else 1e-3
    atol = 5e-1 if self.using_tpu else 1e-2

    # Set up the device ops and run the test.
    computed_norms, true_norms = self.strategy.run(test_op, args=(x_batch,))
    # TPUs return replica contexts, which must be unwrapped.
    if self.using_tpu:
      common_test_utils.assert_replica_values_are_close(self, computed_norms)
      common_test_utils.assert_replica_values_are_close(self, true_norms)
      computed_norms = computed_norms.values[0]
      true_norms = true_norms.values[0]
    expected_size = num_microbatches or batch_size
    self.assertEqual(tf.shape(computed_norms)[0], expected_size)
    self.assertAllClose(computed_norms, true_norms, rtol=rtol, atol=atol)


if __name__ == '__main__':
  tf.test.main()

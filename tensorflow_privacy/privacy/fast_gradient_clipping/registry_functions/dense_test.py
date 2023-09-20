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
from tensorflow_privacy.privacy.fast_gradient_clipping import common_test_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import dense


def get_dense_layer_generators():
  def sigmoid_dense_layer(units):
    return tf.keras.layers.Dense(units, activation='sigmoid')

  return {
      'pure_dense': lambda a, b: tf.keras.layers.Dense(b[0]),
      'sigmoid_dense': lambda a, b: sigmoid_dense_layer(b[0]),
  }


def get_dense_model_generators():
  return {
      'func1': common_test_utils.make_one_layer_functional_model,
      'func2': common_test_utils.make_two_layer_functional_model,
      'tower2': common_test_utils.make_two_tower_model,
  }


def get_dense_layer_registries():
  dense_registry = layer_registry.LayerRegistry()
  dense_registry.insert(tf.keras.layers.Dense, dense.dense_layer_computation)
  return {
      'dense_only': dense_registry,
      'default': layer_registry.make_default_layer_registry(),
  }


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  @parameterized.product(
      model_name=list(get_dense_model_generators().keys()),
      layer_name=list(get_dense_layer_generators().keys()),
      input_dims=[[4]],
      output_dim=[2],
      layer_registry_name=list(get_dense_layer_registries().keys()),
      per_example_loss_fn=[None, common_test_utils.test_loss_fn],
      num_microbatches=[None, 1, 2],
      is_eager=[True, False],
      partial=[True, False],
      weighted=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      model_name,
      layer_name,
      input_dims,
      output_dim,
      layer_registry_name,
      per_example_loss_fn,
      num_microbatches,
      is_eager,
      partial,
      weighted,
  ):
    # Parse inputs to generate test data.
    x_batches, weight_batches = common_test_utils.get_nd_test_batches(
        input_dims[0]
    )

    # Load shared assets to all devices.
    with self.strategy.scope():
      model = common_test_utils.get_model_from_generator(
          model_generator=get_dense_model_generators()[model_name],
          layer_generator=get_dense_layer_generators()[layer_name],
          input_dims=input_dims,
          output_dims=[output_dim],
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op(x_batch, weight_batch):
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=per_example_loss_fn,
          num_microbatches=num_microbatches,
          x_batch=[x_batch, x_batch] if model_name == 'tower2' else x_batch,
          weight_batch=weight_batch if weighted else None,
          registry=get_dense_layer_registries()[layer_registry_name],
          partial=partial,
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, jit_compile=True, autograph=False)

    # TPUs use lower precision than CPUs, so we relax our criterion.
    # E.g., one of the TPU runs generated the following results:
    #
    #   computed_norm = 22.530651
    #   true_norm     = 22.570976
    #   abs_diff      = 0.04032516
    #   rel_diff      = 0.00178659
    #
    # which is a reasonable level of error for computing gradient norms.
    # Other trials also give an absolute (resp. relative) error of around
    # 0.05 (resp. 0.0015).
    rtol = 1e-2 if self.using_tpu else 1e-3
    atol = 1e-1 if self.using_tpu else 1e-2

    for x_batch, weight_batch in zip(x_batches, weight_batches):
      batch_size = x_batch.shape[0]
      if num_microbatches is not None and batch_size % num_microbatches != 0:
        continue
      # Set up the device ops and run the test.
      computed_norms, true_norms = self.strategy.run(
          test_op, args=(x_batch, weight_batch)
      )
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

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
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import dense
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import nlp_position_embedding


def get_nlp_position_embedding_model_generators():
  return {
      'func1': common_test_utils.make_one_layer_functional_model,
  }


def get_nlp_position_embedding_inputs():
  """Generates input_data."""
  # (input_dims, max_length, seq_axis)
  return [
      # Rank-2 Tensors
      ([3, 2], 6, 1),
      ([3, 2], 3, 1),
      # Rank-3 Tensors
      ([4, 3, 2], 8, 1),
      ([4, 3, 2], 4, 1),
      ([4, 3, 2], 6, 2),
      ([4, 3, 2], 3, 2),
  ]


def get_nlp_position_embedding_layer_registries():
  dbl_registry = layer_registry.LayerRegistry()
  dbl_registry.insert(tf.keras.layers.Dense, dense.dense_layer_computation)
  dbl_registry.insert(
      tfm.nlp.layers.PositionEmbedding,
      nlp_position_embedding.nlp_position_embedding_layer_computation,
  )
  return {
      'embed_and_dense': dbl_registry,
  }


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  # TODO(weiweikong): Test sparse input tensors when the GitHub CI environment
  # supports them for embeddings.
  @parameterized.product(
      input_data=get_nlp_position_embedding_inputs(),
      model_name=list(get_nlp_position_embedding_model_generators()),
      layer_registry_name=list(get_nlp_position_embedding_layer_registries()),
      num_microbatches=[None, 2],
      is_eager=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      input_data,
      model_name,
      layer_registry_name,
      num_microbatches,
      is_eager,
  ):
    # Parse inputs to generate test data.
    input_dims, max_length, seq_axis = input_data

    # Load shared assets to all devices.
    with self.strategy.scope():

      def embed_layer_generator(a, b):
        del a, b  # Unused input variables.
        return tfm.nlp.layers.PositionEmbedding(
            max_length=max_length,
            seq_axis=seq_axis,
        )

      batch_size = 6
      dummy_output_dims = [1]
      example_size = tf.reduce_prod(input_dims)
      example_values = tf.range(batch_size * example_size, dtype=tf.float32)
      x_batch = tf.reshape(example_values, [batch_size] + input_dims)
      model = common_test_utils.get_model_from_generator(
          model_generator=(
              get_nlp_position_embedding_model_generators()[model_name]
          ),
          layer_generator=embed_layer_generator,
          input_dims=input_dims,
          output_dims=dummy_output_dims,
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op():
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=None,
          num_microbatches=num_microbatches,
          x_batch=x_batch,
          registry=(
              get_nlp_position_embedding_layer_registries()[layer_registry_name]
          ),
          partial=None,
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, autograph=False)

    # Set up the device ops and run the test.
    computed_norms, true_norms = self.strategy.run(test_op)
    # TPUs return replica contexts, which must be unwrapped.
    if self.using_tpu:
      common_test_utils.assert_replica_values_are_close(self, computed_norms)
      common_test_utils.assert_replica_values_are_close(self, true_norms)
      computed_norms = computed_norms.values[0]
      true_norms = true_norms.values[0]
    expected_size = num_microbatches or batch_size
    self.assertEqual(tf.shape(computed_norms)[0], expected_size)
    self.assertAllClose(computed_norms, true_norms, rtol=1e-3, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()

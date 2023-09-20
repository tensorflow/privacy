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
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import embedding


def get_embedding_model_generators():
  return {
      'bow1': common_test_utils.make_bow_model,
      'bow2': common_test_utils.make_dense_bow_model,
      'weighted_bow1': common_test_utils.make_weighted_bow_model,
  }


def get_embedding_inputs():
  """Generates test pairs of the form (is_ragged, input_data)."""
  return [
      # 2D inputs.
      (False, [[0, 1]]),
      (False, [[0, 1], [1, 1], [0, 0]]),
      (True, [[0], [1], [], [0, 0], [0, 1], [1, 0], [1, 1]]),
      (True, [[0], [1], [], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1]]),
      # 3D inputs.
      (False, [[[0, 1]]]),
      (False, [[[0, 1]], [[1, 1]], [[0, 0]]]),
      (True, [[[0]], [[1]], [], [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]),
      (True, [[[0]], [[1]], [], [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]], [[0]]]),
  ]


def get_embedding_layer_registries():
  dbl_registry = layer_registry.LayerRegistry()
  dbl_registry.insert(tf.keras.layers.Dense, dense.dense_layer_computation)
  dbl_registry.insert(
      tf.keras.layers.Embedding, embedding.embedding_layer_computation
  )
  return {
      'embed_and_dense': dbl_registry,
      'default': layer_registry.make_default_layer_registry(),
  }


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  # TODO(weiweikong): Test sparse input tensors when the GitHub CI environment
  # supports them for embeddings.
  @parameterized.product(
      x_inputs=get_embedding_inputs(),
      model_name=list(get_embedding_model_generators().keys()),
      output_dim=[2],
      layer_registry_name=list(get_embedding_layer_registries().keys()),
      per_example_loss_fn=[None, common_test_utils.test_loss_fn],
      num_microbatches=[None, 2],
      is_eager=[True, False],
      partial=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      x_inputs,
      model_name,
      output_dim,
      layer_registry_name,
      per_example_loss_fn,
      num_microbatches,
      is_eager,
      partial,
  ):
    # Parse inputs to generate test data.
    is_ragged, input_data = x_inputs
    embed_indices = (
        tf.ragged.constant(input_data, dtype=tf.int32)
        if is_ragged
        else tf.convert_to_tensor(input_data, dtype_hint=tf.int32)
    )

    # The following are invalid test combinations and, hence, are skipped.
    batch_size = embed_indices.shape[0]
    if (
        (num_microbatches is not None and batch_size % num_microbatches != 0)
        or (model_name == 'weighted_bow1' and is_ragged)
        or (
            # Current clipping ops do not have corresponding TPU kernels.
            self.using_tpu
            and is_ragged
        )
    ):
      return

    # Load shared assets to all devices.
    with self.strategy.scope():

      def embed_layer_generator(_, output_dims):
        return tf.keras.layers.Embedding(10, *output_dims)

      model = common_test_utils.get_model_from_generator(
          model_generator=get_embedding_model_generators()[model_name],
          layer_generator=embed_layer_generator,
          input_dims=embed_indices.shape[1:],
          output_dims=[output_dim],
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op(x_batch):
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=per_example_loss_fn,
          num_microbatches=num_microbatches,
          x_batch=x_batch,
          registry=get_embedding_layer_registries()[layer_registry_name],
          partial=partial,
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, autograph=False)

    # Set up the device ops and run the test.
    computed_norms, true_norms = self.strategy.run(
        test_op, args=(embed_indices,)
    )
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

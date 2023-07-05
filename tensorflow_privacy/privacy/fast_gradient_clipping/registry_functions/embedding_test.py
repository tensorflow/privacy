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


# ==============================================================================
# Helper functions.
# ==============================================================================
def get_embedding_model_generators():
  return {
      'bow1': common_test_utils.make_bow_model,
      'bow2': common_test_utils.make_dense_bow_model,
      'weighted_bow1': common_test_utils.make_weighted_bow_model,
  }


# ==============================================================================
# Main tests.
# ==============================================================================
class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  # TODO(weiweikong): Test sparse input tensors when the GitHub CI environment
  # supports them for embeddings.
  @parameterized.product(
      x_batch=[
          # 2D inputs.
          tf.convert_to_tensor([[0, 1]], dtype_hint=tf.int32),
          tf.convert_to_tensor([[0, 1], [1, 1], [0, 0]], dtype_hint=tf.int32),
          tf.ragged.constant(
              [[0], [1], [], [0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.int32
          ),
          tf.ragged.constant(
              [[0], [1], [], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1]],
              dtype=tf.int32,
          ),
          # 3D inputs.
          tf.convert_to_tensor([[[0, 1]]], dtype_hint=tf.int32),
          tf.convert_to_tensor(
              [[[0, 1]], [[1, 1]], [[0, 0]]], dtype_hint=tf.int32
          ),
          tf.ragged.constant(
              [[[0]], [[1]], [], [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]],
              dtype=tf.int32,
          ),
          tf.ragged.constant(
              [[[0]], [[1]], [], [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]], [[0]]],
              dtype=tf.int32,
          ),
      ],
      model_name=list(get_embedding_model_generators().keys()),
      output_dim=[2],
      per_example_loss_fn=[None, common_test_utils.test_loss_fn],
      num_microbatches=[None, 2],
      is_eager=[True, False],
      partial=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      x_batch,
      model_name,
      output_dim,
      per_example_loss_fn,
      num_microbatches,
      is_eager,
      partial,
  ):
    batch_size = x_batch.shape[0]
    # The following are invalid test combinations, and are skipped.
    if (
        num_microbatches is not None and batch_size % num_microbatches != 0
    ) or (
        model_name == 'weighted_bow1' and isinstance(x_batch, tf.RaggedTensor)
    ):
      return
    default_registry = layer_registry.make_default_layer_registry()
    model_generator = get_embedding_model_generators()[model_name]
    computed_norms, true_norms = (
        common_test_utils.get_computed_and_true_norms(
            model_generator=model_generator,
            layer_generator=None,
            input_dims=x_batch.shape[1:],
            output_dim=output_dim,
            per_example_loss_fn=per_example_loss_fn,
            num_microbatches=num_microbatches,
            is_eager=is_eager,
            x_batch=x_batch,
            registry=default_registry,
            partial=partial,
        )
    )
    self.assertEqual(computed_norms.shape[0], num_microbatches or batch_size)
    self.assertAllClose(computed_norms, true_norms, rtol=1e-3, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()

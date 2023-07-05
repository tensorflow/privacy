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
def get_dense_layer_generators():
  def sigmoid_dense_layer(b):
    return tf.keras.layers.Dense(b, activation='sigmoid')

  return {
      'pure_dense': lambda a, b: tf.keras.layers.Dense(b),
      'sigmoid_dense': lambda a, b: sigmoid_dense_layer(b),
  }


def get_dense_model_generators():
  return {
      'seq1': common_test_utils.make_two_layer_sequential_model,
      'seq2': common_test_utils.make_three_layer_sequential_model,
      'func1': common_test_utils.make_two_layer_functional_model,
      'tower1': common_test_utils.make_two_tower_model,
  }


# ==============================================================================
# Main tests.
# ==============================================================================
class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      model_name=list(get_dense_model_generators().keys()),
      layer_name=list(get_dense_layer_generators().keys()),
      input_dim=[4],
      output_dim=[2],
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
      input_dim,
      output_dim,
      per_example_loss_fn,
      num_microbatches,
      is_eager,
      partial,
      weighted,
  ):
    model_generator = get_dense_model_generators()[model_name]
    layer_generator = get_dense_layer_generators()[layer_name]
    x_batches, weight_batches = common_test_utils.get_nd_test_batches(input_dim)
    default_registry = layer_registry.make_default_layer_registry()
    for x_batch, weight_batch in zip(x_batches, weight_batches):
      batch_size = x_batch.shape[0]
      if num_microbatches is not None and batch_size % num_microbatches != 0:
        continue
      computed_norms, true_norms = (
          common_test_utils.get_computed_and_true_norms(
              model_generator,
              layer_generator,
              input_dim,
              output_dim,
              per_example_loss_fn,
              num_microbatches,
              is_eager,
              x_batch=[x_batch, x_batch] if model_name == 'tower1' else x_batch,
              weight_batch=weight_batch if weighted else None,
              registry=default_registry,
              partial=partial,
          )
      )
      expected_size = num_microbatches or batch_size
      self.assertEqual(computed_norms.shape[0], expected_size)
      self.assertAllClose(computed_norms, true_norms, rtol=1e-3, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()

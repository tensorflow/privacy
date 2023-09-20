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
from tensorflow_privacy.privacy.fast_gradient_clipping import common_test_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import dense
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import layer_normalization


def get_layer_norm_layer_generators():
  return {
      'defaults': lambda x: tf.keras.layers.LayerNormalization(axis=x),
  }


def get_layer_norm_model_generators():
  return {
      # TODO(b/274483956): Test more complex models once the we can support
      # `nD` inputs for `tf.keras.layers.Dense`.
      'func1': common_test_utils.make_one_layer_functional_model,
  }


def get_layer_norm_parameter_tuples():
  """Consists of (input_dims, parameter_axes)."""
  return [
      # Rank-2
      ([3], -1),
      ([3], [1]),
      # Rank-3
      ([3, 4], -1),
      ([3, 4], [1]),
      ([3, 4], [2]),
      ([3, 4], [1, 2]),
      # Rank-4
      ([3, 4, 5], -1),
      ([3, 4, 5], [1]),
      ([3, 4, 5], [2]),
      ([3, 4, 5], [3]),
      ([3, 4, 5], [1, 2]),
      ([3, 4, 5], [1, 3]),
      ([3, 4, 5], [2, 3]),
      ([3, 4, 5], [1, 2, 3]),
  ]


def get_layer_norm_registries():
  ln_registry = layer_registry.LayerRegistry()
  ln_registry.insert(tf.keras.layers.Dense, dense.dense_layer_computation)
  ln_registry.insert(
      tf.keras.layers.LayerNormalization,
      layer_normalization.layer_normalization_computation,
  )
  return {
      'layer_norm_only': ln_registry,
  }


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  @parameterized.product(
      model_name=list(get_layer_norm_model_generators().keys()),
      layer_name=list(get_layer_norm_layer_generators().keys()),
      parameter_tuple=get_layer_norm_parameter_tuples(),
      layer_registry_name=list(get_layer_norm_registries().keys()),
      num_microbatches=[None, 2],
      is_eager=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      model_name,
      layer_name,
      parameter_tuple,
      layer_registry_name,
      num_microbatches,
      is_eager,
  ):
    # Parse inputs to generate test data.
    input_dims, parameter_axes = parameter_tuple

    def curried_generator(a, b):
      del a, b  # Unused by the generator.
      layer_norm_generator = get_layer_norm_layer_generators()[layer_name]
      return layer_norm_generator(parameter_axes)

    # Load shared assets to all devices.
    with self.strategy.scope():
      dummy_output_dim = 1
      model = common_test_utils.get_model_from_generator(
          model_generator=get_layer_norm_model_generators()[model_name],
          layer_generator=curried_generator,
          input_dims=input_dims,
          output_dims=[dummy_output_dim],
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op(x_batch):
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=None,
          num_microbatches=num_microbatches,
          x_batch=[x_batch, x_batch] if model_name == 'tower2' else x_batch,
          weight_batch=None,
          registry=get_layer_norm_registries()[layer_registry_name],
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, jit_compile=True, autograph=False)

    # TPUs use lower precision than CPUs, so we relax our criterion (see
    # `dense_test.py` for additional discussions).
    rtol = 1e-2 if self.using_tpu else 1e-3
    atol = 1e-1 if self.using_tpu else 1e-2

    # Each batched input is a reshape of a `tf.range()` call.
    batch_size = 2
    example_size = np.prod(input_dims)
    example_values = tf.range(batch_size * example_size, dtype=tf.float32)
    x_batch = tf.reshape(example_values, [batch_size] + input_dims)
    batch_size = x_batch.shape[0]
    # Set up the device ops and run the test.
    computed_norms, true_norms = self.strategy.run(test_op, args=(x_batch,))
    # TPUs return replica contexts, which must be unwrapped.
    if self.using_tpu:
      common_test_utils.assert_replica_values_are_close(self, computed_norms)
      common_test_utils.assert_replica_values_are_close(self, true_norms)
      computed_norms = computed_norms.values[0]
      true_norms = true_norms.values[0]
    self.assertEqual(tf.shape(computed_norms)[0], batch_size)
    self.assertAllClose(computed_norms, true_norms, rtol=rtol, atol=atol)


if __name__ == '__main__':
  tf.test.main()

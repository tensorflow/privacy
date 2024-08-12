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

from typing import Any

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.sparsity_preserving_noise import layer_registry as snlr


# ==============================================================================
# Helper functions and classes.
# ==============================================================================
@tf.keras.utils.register_keras_serializable('gradient_clipping_utils_test')
class DoubleDense(tf.keras.layers.Layer):
  """Generates two dense layers nested together."""

  def __init__(self, units: int):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(units, name='DDense_ext_1')
    self.dense2 = tf.keras.layers.Dense(1, name='DDense_ext_2')

  def call(self, inputs: Any):
    x = self.dense1(inputs)
    return self.dense2(x)


@tf.keras.utils.register_keras_serializable('gradient_clipping_utils_test')
class TripleDense(tf.keras.layers.Layer):
  """Generates three dense layers nested together."""

  def __init__(self, units: int):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(units, name='TDense_ext_1')
    self.dense2 = tf.keras.layers.Dense(units, name='TDense_ext_2')
    self.dense3 = tf.keras.layers.Dense(1, name='TDense_ext_3')

  def call(self, inputs: Any):
    x1 = self.dense1(inputs)
    x2 = self.dense2(x1)
    return self.dense3(x2)


def get_reduced_model(sample_inputs, hidden_layer_list, new_custom_layers=None):
  """Reduces a set of layers to only core Keras layers in a model."""
  sample_outputs = sample_inputs
  for l in hidden_layer_list:
    sample_outputs = l(sample_outputs)
  custom_model = tf.keras.Model(inputs=sample_inputs, outputs=sample_outputs)
  if new_custom_layers:
    reduced_outputs = (
        gradient_clipping_utils.generate_model_outputs_using_core_keras_layers(
            custom_model,
            custom_layer_set=new_custom_layers,
        )
    )
  else:
    reduced_outputs = (
        gradient_clipping_utils.generate_model_outputs_using_core_keras_layers(
            custom_model
        )
    )
  return tf.keras.Model(inputs=custom_model.inputs, outputs=reduced_outputs)


# ==============================================================================
# Main tests.
# ==============================================================================
class ModelForwardPassTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      input_packing_type=[None, tuple, list, dict],
      output_packing_type=[None, tuple, list, dict],
  )
  def test_outputs_are_consistent(
      self, input_packing_type, output_packing_type
  ):
    num_dims = 3
    num_inputs = 1 if input_packing_type is None else 2
    num_outputs = 1 if output_packing_type is None else 2
    sample_inputs = [tf.keras.Input((num_dims,)) for _ in range(num_inputs)]
    temp_sum = tf.stack(sample_inputs, axis=0)
    sample_outputs = [
        tf.multiply(temp_sum, float(i + 1.0)) for i in range(num_outputs)
    ]
    sample_x_batch = [
        tf.multiply(tf.range(num_dims, dtype=tf.float32), float(i + 1.0))
        for i in range(num_inputs)
    ]

    # Pack inputs.
    if input_packing_type is None:
      inputs = sample_inputs[0]
      x_batch = sample_x_batch[0]
    elif input_packing_type is not dict:
      inputs = input_packing_type(sample_inputs)
      x_batch = input_packing_type(sample_x_batch)
    else:
      inputs = {}
      x_batch = {}
      keys = [str(i) for i in range(len(sample_inputs))]
      for k, v1, v2 in zip(keys, sample_inputs, sample_x_batch):
        inputs[k] = v1
        x_batch[k] = v2

    # Pack outputs.
    if output_packing_type is None:
      outputs = sample_outputs[0]
    elif output_packing_type is not dict:
      outputs = output_packing_type(sample_outputs)
    else:
      outputs = {}
      keys = [str(i) for i in range(len(sample_outputs))]
      for k, v in zip(keys, sample_outputs):
        outputs[k] = v

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    computed_outputs, _ = gradient_clipping_utils.model_forward_pass(
        model,
        x_batch,
    )
    true_outputs = model(x_batch)
    self.assertAllClose(computed_outputs, true_outputs)


class GenerateOutputsUsingCoreKerasLayers(
    tf.test.TestCase, parameterized.TestCase
):

  def test_single_custom_layer_is_reduced(self):
    num_units = 5
    num_dims = 3
    reduced_model = get_reduced_model(
        tf.keras.Input(num_dims),
        [DoubleDense(num_units)],
    )
    # Ignore the input layer.
    for l in reduced_model.layers[1:]:
      self.assertIsInstance(l, tf.keras.layers.Dense)

  def test_two_distinct_custom_layers_are_reduced(self):
    num_units = 5
    num_dims = 3
    reduced_model = get_reduced_model(
        tf.keras.Input(num_dims),
        [DoubleDense(num_units), TripleDense(num_units)],
    )
    # Ignore the input layer.
    for l in reduced_model.layers[1:]:
      self.assertIsInstance(l, tf.keras.layers.Dense)

  def test_new_custom_layer_spec(self):
    num_units = 5
    num_dims = 3
    reduced_model = get_reduced_model(
        tf.keras.Input(num_dims),
        [DoubleDense(num_units), TripleDense(num_units)],
        new_custom_layers=set([DoubleDense]),
    )
    # Ignore the input layer.
    for l in reduced_model.layers[1:]:
      self.assertTrue(
          isinstance(l, tf.keras.layers.Dense) or isinstance(l, TripleDense)
      )


class RegistryGeneratorFnTest(tf.test.TestCase, parameterized.TestCase):

  def _get_sparse_layer_registry(self):
    def count_contribution_fn(_):
      return None

    def registry_fn(*_):
      return {'var': count_contribution_fn}

    registry = snlr.LayerRegistry()
    registry.insert(tf.keras.layers.Embedding, registry_fn)
    return registry, count_contribution_fn

  def _get_layer_registry(self):
    var = tf.Variable(1.0)
    output = tf.ones((1, 1))

    def sqr_norm_fn(_):
      return None

    def registry_fn(*_):
      return [var], output, sqr_norm_fn

    registry = lr.LayerRegistry()
    registry.insert(tf.keras.layers.Embedding, registry_fn)
    registry.insert(tf.keras.layers.Dense, registry_fn)
    return registry, var, output, sqr_norm_fn

  def test_registry_generator_fn(self):
    inputs = tf.constant([[0, 1]])
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10, 1),
        tf.keras.layers.Dense(1),
    ])

    sparse_layer_registry, count_contribution_fn = (
        self._get_sparse_layer_registry()
    )
    layer_registry, var, output, sqr_norm_fn = self._get_layer_registry()
    registry_generator_fn = gradient_clipping_utils.get_registry_generator_fn(
        tape=tf.GradientTape(),
        layer_registry=layer_registry,
        sparse_noise_layer_registry=sparse_layer_registry,
        num_microbatches=None,
    )
    embedding_layer = model.layers[0]
    out, embedding_registry_generator_fn_output = registry_generator_fn(
        embedding_layer,
        [inputs],
        {},
    )
    expected_embedding_registry_generator_fn_output = (
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id=str(id(embedding_layer)),
            layer_vars=[var],
            layer_sqr_norm_fn=sqr_norm_fn,
            varname_to_count_contribution_fn={'var': count_contribution_fn},
            layer_trainable_weights=embedding_layer.trainable_weights,
        )
    )
    self.assertEqual(
        embedding_registry_generator_fn_output,
        expected_embedding_registry_generator_fn_output,
    )
    self.assertEqual(out, output)
    dense_layer = model.layers[1]
    out, dense_registry_generator_fn_output = registry_generator_fn(
        dense_layer,
        [inputs],
        {},
    )
    expected_dense_registry_generator_fn_output = (
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id=str(id(dense_layer)),
            layer_vars=[var],
            layer_sqr_norm_fn=sqr_norm_fn,
            varname_to_count_contribution_fn=None,
            layer_trainable_weights=dense_layer.trainable_weights,
        )
    )
    self.assertEqual(
        dense_registry_generator_fn_output,
        expected_dense_registry_generator_fn_output,
    )
    self.assertEqual(out, output)


if __name__ == '__main__':
  tf.test.main()

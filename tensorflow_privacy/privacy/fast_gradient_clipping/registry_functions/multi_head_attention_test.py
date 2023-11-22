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
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import multi_head_attention


def get_attention_layer_generators():
  def basic_attention_layer(
      num_heads,
      key_dim,
      value_dim,
      dropout,
      use_bias,
      output_shape,
  ):
    return tf.keras.layers.MultiHeadAttention(
        num_heads,
        key_dim,
        value_dim=value_dim,
        dropout=dropout,
        use_bias=use_bias,
        output_shape=output_shape,
    )

  return {
      'basic_attention_layer': basic_attention_layer,
  }


def make_one_layer_attention_model(layer_generator, input_dims, output_dims):
  """Creates a 1-layer MultiHeadAttention model."""
  inputs, input_args, input_kwargs = get_multi_head_attention_model_inputs(
      input_dims
  )
  layer1 = layer_generator(input_dims, output_dims)
  del output_dims
  temp1 = layer1(*input_args, **input_kwargs)
  outputs = common_test_utils.reshape_and_sum(temp1)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_two_layer_attention_model(layer_generator, input_dims, output_dims):
  """Creates a 2-layer MultiHeadAttention model."""
  inputs, input_args, input_kwargs = get_multi_head_attention_model_inputs(
      input_dims
  )
  layer1 = layer_generator(input_dims, output_dims)
  temp1 = layer1(*input_args, **input_kwargs)
  temp2 = tf.keras.layers.Dense(1)(temp1)
  outputs = common_test_utils.reshape_and_sum(temp2)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_attention_model_generators():
  return {
      'seq1_mha': make_one_layer_attention_model,
      'seq2_mha': make_two_layer_attention_model,
  }


def get_attention_parameter_tuples():
  # (query_input_dims, value_input_dims, use_key, use_attention_mask,
  #  num_heads, key_dim, value_dim, dropout, use_bias, output_shape)
  return [
      # Small instances, default flags.
      ([2, 3], [3, 4], False, False, 2, 2, 3, 0.0, True, None),  # defaults
      ([2, 3], [3, 4], True, False, 2, 2, 3, 0.0, True, None),  # use key
      ([2, 3], [3, 4], False, False, 2, 2, 3, 0.0, False, None),  # no bias
      ([2, 3], [3, 4], False, False, 2, 2, 3, 0.1, True, None),  # dropout
      ([2, 3], [3, 4], False, False, 2, 2, 3, 0.0, True, [3]),  # output shape
      ([2, 3], [3, 4], False, False, 1, 2, 3, 0.0, True, 3),  # single head
      ([2, 3], [3, 4], False, True, 2, 2, 3, 0.0, True, None),  # attention mask
  ]


def get_attention_layer_registries():
  attention_and_dense = layer_registry.LayerRegistry()
  attention_and_dense.insert(
      tf.keras.layers.MultiHeadAttention,
      multi_head_attention.multi_head_attention_layer_computation,
  )
  attention_and_dense.insert(
      tf.keras.layers.Dense,
      dense.dense_layer_computation,
  )
  return {
      'attention_and_dense': attention_and_dense,
  }


def get_multi_head_attention_example_inputs(
    query_input_dims,
    value_input_dims,
    ragged_key=False,
    ragged_value=False,
    ragged_query=False,
):
  """Generates example MultiHeadAttention concrete inputs for testing."""
  # Each batched input is a reshape of a `tf.range()` call.
  batch_size = 2
  # Query input tensor.
  query_size = tf.reduce_prod(query_input_dims)
  query_tsr = tf.range(batch_size * query_size, dtype=tf.float32) / tf.cast(
      query_size, tf.float32
  )
  query_batch = tf.reshape(query_tsr, [batch_size] + query_input_dims)
  # Value input tensor.
  value_size = tf.reduce_prod(value_input_dims)
  value_tsr = (
      2.0
      * tf.range(batch_size * value_size, dtype=tf.float32)
      / tf.cast(value_size, tf.float32)
  )
  value_batch = tf.reshape(value_tsr, [batch_size] + value_input_dims)
  # Key input tensor (optional).
  key_tsr = (
      3.0
      * tf.range(batch_size * value_size, dtype=tf.float32)
      / tf.cast(value_size, tf.float32)
  )
  key_batch = tf.reshape(key_tsr, [batch_size] + value_input_dims)
  # Attention mask input tensor (optional).
  mask_size = tf.reduce_prod(query_input_dims[:-1]) * tf.reduce_prod(
      value_input_dims[:-1]
  )
  mask_input_dims = query_input_dims[:-1] + value_input_dims[:-1]
  mask_tsr = tf.random.uniform([int(batch_size * mask_size)]) <= 0.5
  mask_batch = tf.reshape(mask_tsr, [batch_size] + mask_input_dims)
  # Convert to ragged if needed.
  if ragged_query:
    query_batch = tf.RaggedTensor.from_tensor(query_batch)
  if ragged_value:
    value_batch = tf.RaggedTensor.from_tensor(value_batch)
  if ragged_key:
    key_batch = tf.RaggedTensor.from_tensor(key_batch)
  return query_batch, value_batch, key_batch, mask_batch


def get_multi_head_attention_model_inputs(input_dims):
  """Creates MultiHeadAttention symbolic model input."""
  (
      query_input_dims,
      value_input_dims,
      key_input_dims,
      mask_input_dims,
      use_key,
      use_mask,
  ) = input_dims
  query_input = tf.keras.Input(shape=query_input_dims)
  value_input = tf.keras.Input(shape=value_input_dims)
  key_input = tf.keras.Input(shape=key_input_dims)
  mask_input = tf.keras.Input(shape=mask_input_dims)

  input_args = (query_input, value_input)
  input_kwargs = {}
  if use_key:
    input_kwargs['key'] = key_input
  if use_mask:
    input_kwargs['attention_mask'] = mask_input
  return input_args + (key_input, mask_input), input_args, input_kwargs


class CheckOutputs(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      use_key=[True, False],
      use_query_as_kwarg=[True, False],
      use_value_as_kwarg=[True, False],
      use_attention_mask=[True, False],
      return_scores=[True, False],
      ragged_key=[True, False],
      ragged_value=[True, False],
      ragged_query=[True, False],
  )
  def test_verify_consistent_outputs(
      self,
      use_key,
      use_query_as_kwarg,
      use_value_as_kwarg,
      use_attention_mask,
      return_scores,
      ragged_key,
      ragged_value,
      ragged_query,
  ):
    num_heads = 2
    key_dim, value_dim = (2, 3)
    query_input_dims = [2, 3]
    value_input_dims = [3, 4]
    query_batch, value_batch, key_batch, mask_batch = (
        get_multi_head_attention_example_inputs(
            query_input_dims,
            value_input_dims,
            ragged_key=ragged_key,
            ragged_value=ragged_value,
            ragged_query=ragged_query,
        )
    )
    layer_instance = tf.keras.layers.MultiHeadAttention(
        num_heads, key_dim, value_dim
    )

    # Set up test inputs, branched on input order.
    input_kwargs = {
        'key': key_batch if use_key else None,
        'attention_mask': mask_batch if use_attention_mask else None,
        'return_attention_scores': return_scores,
    }
    input_args = tuple()
    if use_value_as_kwarg and use_query_as_kwarg:
      input_kwargs['query'] = query_batch
      input_kwargs['value'] = value_batch
    elif use_value_as_kwarg:
      input_kwargs['value'] = value_batch
      input_args = (query_batch,)
    elif use_query_as_kwarg:
      # Invalid test case; cannot pass query kwarg after value.
      return
    else:
      input_args = (query_batch, key_batch)

    dummy_tape = tf.GradientTape()
    with dummy_tape:
      _, computed_outputs, _ = (
          multi_head_attention.multi_head_attention_layer_computation(
              layer_instance=layer_instance,
              input_args=input_args,
              input_kwargs=input_kwargs,
              tape=dummy_tape,
          )
      )
    true_outputs = layer_instance(*input_args, **input_kwargs)
    self.assertAllClose(computed_outputs, true_outputs)


class GradNormTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = tf.distribute.get_strategy()
    self.using_tpu = False

  @parameterized.product(
      model_name=list(get_attention_model_generators().keys()),
      layer_name=list(get_attention_layer_generators().keys()),
      layer_registry_name=list(get_attention_layer_registries().keys()),
      param_tuple=get_attention_parameter_tuples(),
      num_microbatches=[None, 2],
      is_eager=[True, False],
  )
  def test_gradient_norms_on_various_models(
      self,
      model_name,
      layer_name,
      layer_registry_name,
      param_tuple,
      num_microbatches,
      is_eager,
  ):
    # Parse inputs to generate test data.
    (
        query_input_dims,
        value_input_dims,
        use_key,
        use_attention_mask,
        num_heads,
        key_dim,
        value_dim,
        dropout,
        use_bias,
        output_shape,
    ) = param_tuple
    attention_generator_inputs = (
        num_heads,
        key_dim,
        value_dim,
        dropout,
        use_bias,
        output_shape,
    )
    query_batch, value_batch, key_batch, mask_batch = (
        get_multi_head_attention_example_inputs(
            query_input_dims, value_input_dims
        )
    )
    mask_input_dims = query_input_dims[:-1] + value_input_dims[:-1]

    # Make the layer generator via currying.
    attention_generator = get_attention_layer_generators()[layer_name]

    def curried_generator(a, b):
      del a, b
      return attention_generator(*attention_generator_inputs)

    # Load shared assets to all devices.
    with self.strategy.scope():
      model = common_test_utils.get_model_from_generator(
          model_generator=get_attention_model_generators()[model_name],
          layer_generator=curried_generator,
          input_dims=(
              query_input_dims,
              value_input_dims,
              value_input_dims,
              mask_input_dims,
              use_key,
              use_attention_mask,
          ),
          output_dims=None,
          is_eager=is_eager,
      )

    # Define the main testing ops. These may be later compiled to a Graph op.
    def test_op(query_batch, value_batch, key_batch, mask_batch):
      return common_test_utils.get_computed_and_true_norms_from_model(
          model=model,
          per_example_loss_fn=None,
          num_microbatches=num_microbatches,
          x_batch=(query_batch, value_batch, key_batch, mask_batch),
          weight_batch=None,
          registry=get_attention_layer_registries()[layer_registry_name],
          partial=False,
      )

    # TPUs can only run `tf.function`-decorated functions.
    if self.using_tpu:
      test_op = tf.function(test_op, jit_compile=True, autograph=False)

    # TPUs use lower precision than CPUs, so we relax our criterion.
    # E.g., one of the TPU runs generated the following results:
    #
    #   computed_norm = 22.756414
    #   true_norm     = 23.338600
    #   abs_diff      = 0.58218575
    #   rel_diff      = 0.02494519
    #
    # which is a reasonable level of error for computing gradient norms.
    # Other trials also give an absolute (resp. relative) error of around
    # 0.05 (resp. 0.0015).
    rtol = 1e-1 if self.using_tpu else 1e-2
    atol = 1e-0 if self.using_tpu else 1e-1

    # Set up the device ops and run the test.
    computed_norms, true_norms = self.strategy.run(
        test_op, args=(query_batch, value_batch, key_batch, mask_batch)
    )
    # TPUs return replica contexts, which must be unwrapped.
    batch_size = tf.shape(query_batch)[0]
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

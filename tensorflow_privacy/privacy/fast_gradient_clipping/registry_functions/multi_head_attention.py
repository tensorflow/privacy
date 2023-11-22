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
"""Fast clipping function for `tf.keras.layers.MultiHeadAttention`."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import einsum_dense


def multi_head_attention_layer_computation(
    layer_instance: tf.keras.layers.MultiHeadAttention,
    input_args: Sequence[Any],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.MultiHeadAttention`.

  This function essentially applies the registry function for
  `tf.keras.layers.EinsumDense` three times. Some hints about the nature of
  the Einsum transforms are given below.

  -------------------
  ABOUT INPUT SHAPES
  -------------------
  For a given {query, key, value} input `I` of shape

    [Eq. A]  tf.shape(I) == [n, a[0],... , a[k-1], b]

  where `n` is the batch size, the corresponding Einsum equation for its
  `EinsumDense` transform is given by:

    {n a[0] ... a[k-1] b},{b c d}->{n a[1] ... a[k-1] c d}

  where `c` corresponds to the number of attention heads
  (`layer_instance.num_heads`) and `d` corresponds to the size per head
  (`layer_instance.key_dim` or `layer_instance.value_dim`).

  It is expected that the rank of the query, key, and value inputs are the same.

  ------------------
  ABOUT OUTPUT SHAPE
  ------------------
  Suppose the shape of the `query` input `Q` is given by [Eq. A] above with
  `I == Q`. Then, if `layer_instance.output_shape is None`, the output `O` of
  the layer satisfies `tf.shape(Q) == tf.shape(O)`. However, if we have
  `layer_instance.output_shape is not None`, then

    tf.shape(Q) == [n, a[0], ..., a[k-1], *layer_instance.output_shape]

  Args:
    layer_instance: A `tf.keras.layers.MultiHeadAttention` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  # ----------------------
  # PREPROCESS THE INPUTS.
  # ----------------------
  query = (
      input_kwargs.get("query")
      if input_kwargs.get("query") is not None
      else input_args[0]
  )
  value = (
      input_kwargs.get("value")
      if input_kwargs.get("value") is not None
      else input_args[1]
  )
  key = input_kwargs.get("key")
  attention_mask = input_kwargs.get("attention_mask")
  return_attention_scores = input_kwargs.get("return_attention_scores")
  training = input_kwargs.get("training")
  use_causal_mask = input_kwargs.get("use_causal_mask")
  attention_mask = layer_instance._compute_attention_mask(  # pylint: disable=protected-access
      query,
      value,
      key=key,
      attention_mask=attention_mask,
      use_causal_mask=use_causal_mask,
  )
  if not layer_instance._built_from_signature:  # pylint: disable=protected-access
    layer_instance._build_from_signature(query=query, value=value, key=key)  # pylint: disable=protected-access
  if key is None:
    key = value

  query_lengths = 0
  query_is_ragged = isinstance(query, tf.RaggedTensor)
  if query_is_ragged:
    query_lengths = query.nested_row_lengths()
    query = query.to_tensor()

  key_is_ragged = isinstance(key, tf.RaggedTensor)
  value_is_ragged = isinstance(value, tf.RaggedTensor)
  if key_is_ragged and value_is_ragged:
    bounding_shape = tf.math.maximum(
        key.bounding_shape(), value.bounding_shape()
    )
    key = key.to_tensor(shape=bounding_shape)
    value = value.to_tensor(shape=bounding_shape)
  elif key_is_ragged:
    key = key.to_tensor(shape=tf.shape(value))
  elif value_is_ragged:
    value = value.to_tensor(shape=tf.shape(key))
  else:
    pass
  # ------------------------------
  # APPLY THE FAST CLIPPING TRICK.
  # ------------------------------
  # trainable_op: W_q * QUERY
  query_base_vars, query, query_sqr_norm_fn = (
      einsum_dense.einsum_layer_computation(
          layer_instance._query_dense,  # pylint: disable=protected-access
          (query,),
          {},
          tape,
          num_microbatches,
      )
  )
  # trainable_op: W_k * KEY
  key_base_vars, key, key_sqr_norm_fn = einsum_dense.einsum_layer_computation(
      layer_instance._key_dense,  # pylint: disable=protected-access
      (key,),
      {},
      tape,
      num_microbatches,
  )
  # trainable_op: W_v * VALUE
  value_base_vars, value, value_sqr_norm_fn = (
      einsum_dense.einsum_layer_computation(
          layer_instance._value_dense,  # pylint: disable=protected-access
          (value,),
          {},
          tape,
          num_microbatches,
      )
  )
  # op: TEMP = ATTENTION(W_q * QUERY, W_k * KEY, W_v * VALUE)
  temp_output, attention_scores = layer_instance._compute_attention(  # pylint: disable=protected-access
      query,
      key,
      value,
      attention_mask,
      training,
  )
  # trainable_op: W_o * OUTPUT
  (
      attention_output_base_vars,
      attention_output,
      attention_output_sqr_norm_fn,
  ) = einsum_dense.einsum_layer_computation(
      layer_instance._output_dense,  # pylint: disable=protected-access
      (temp_output,),
      {},
      tape,
      num_microbatches,
  )
  # ------------------------
  # POSTPROCESS THE OUTPUTS.
  # ------------------------
  # Get registry output tensors ready.
  if query_is_ragged:
    attention_output = tf.RaggedTensor.from_tensor(
        attention_output, query_lengths
    )
  outputs = attention_output
  if return_attention_scores:
    outputs = (attention_output, attention_scores)
  base_vars = [
      query_base_vars,
      key_base_vars,
      value_base_vars,
      attention_output_base_vars,
  ]

  # The square norm function should just aggregate the squared norms
  # corresponding to each trainable op.
  def sqr_norm_fn(grad_list):
    if len(grad_list) != 4:
      raise ValueError(
          "Expected a container of 4 gradients for the `MultiheadAttention` "
          "square norm function's input. Instead, received a container of "
          "size "
          + str(len(grad_list))
      )
    combined_sqr_norms = tf.stack(
        [
            query_sqr_norm_fn(grad_list[0]),
            key_sqr_norm_fn(grad_list[1]),
            value_sqr_norm_fn(grad_list[2]),
            attention_output_sqr_norm_fn(grad_list[3]),
        ],
        axis=1,
    )
    return tf.reduce_sum(combined_sqr_norms, axis=1)

  return base_vars, outputs, sqr_norm_fn

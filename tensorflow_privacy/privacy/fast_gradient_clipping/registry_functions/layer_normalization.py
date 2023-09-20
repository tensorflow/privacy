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
"""Fast clipping function for `tf.keras.layers.LayerNormalization`."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def layer_normalization_computation(
    layer_instance: tf.keras.layers.LayerNormalization,
    input_args: Sequence[Any],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.LayerNormalization`.

  This function computes actual per-example gradients and computes their
  norms directly, instead of employing a chain-rule trick. This is done using
  some slick reshaping calls.

  Args:
    layer_instance: A `tf.keras.layers.LayerNormalization` instance.
    input_args: See `dense_layer_computation()` in `dense.py`.
    input_kwargs: See `dense_layer_computation()` in `dense.py`.
    tape: See `dense_layer_computation()` in `dense.py`.
    num_microbatches: See `dense_layer_computation()` in `dense.py`.

  Returns:
    See `dense_layer_computation()` in `dense.py`.
  """
  del input_kwargs  # Unused in layer normaliztion calls.
  # To make sure the watched variables (beta, gamma) generate per-example
  # gradients, we need to convert trainable variables from shape [S] to
  # [batch_size, S] via duplication to `tf.shape(inputs)` via broadcasting.
  inputs = input_args[0]
  base_vars = []
  batch_size = tf.shape(inputs)[0]

  def process_variable(var):
    """Expand univariate `var` and the expanded tensor to `base_vars`."""
    expanded_var = tf.tile(
        tf.expand_dims(var, axis=0), [batch_size] + [1] * len(var.shape)
    )
    tape.watch(expanded_var)
    base_vars.append(expanded_var)
    broadcast_shape = [1] * len(inputs.shape)
    broadcast_shape[0] = batch_size
    for d in layer_instance.axis:
      broadcast_shape[d] = tf.shape(inputs)[d]
    final_var = tf.reshape(expanded_var, broadcast_shape)
    return final_var

  orig_gamma = layer_instance.gamma
  orig_beta = layer_instance.beta
  layer_instance.gamma = process_variable(orig_gamma)
  layer_instance.beta = process_variable(orig_beta)

  # Do the computation, ensure that the output conforms to the unexpanded
  # computation, and restore the state of the original instance.
  outputs = layer_instance.call(inputs)
  layer_instance.gamma = orig_gamma
  layer_instance.beta = orig_beta

  def sqr_norm_fn(grads):
    stacked_grads = tf.stack(grads, axis=-1)
    if num_microbatches is not None:
      stacked_grads = common_manip_utils.maybe_add_microbatch_axis(
          grads, num_microbatches
      )
      stacked_grads = tf.reduce_sum(stacked_grads, axis=1)
    reduction_axes = tf.range(1, tf.rank(stacked_grads))
    return tf.reduce_sum(tf.square(stacked_grads), axis=reduction_axes)

  return base_vars, outputs, sqr_norm_fn

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
"""Fast clipping function for `tfm.nlp.layers.OnDeviceEmbedding`."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def nlp_position_embedding_layer_computation(
    layer_instance: tf.keras.layers.Layer,
    input_args: Sequence[Any],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tfm.nlp.layers.PositionEmbedding`.

  Args:
    layer_instance: A `tfm.nlp.layers.PositionEmbedding` instance.
    input_args: See `dense_layer_computation()` in `dense.py`.
    input_kwargs: See `dense_layer_computation()` in `dense.py`.
    tape: See `dense_layer_computation()` in `dense.py`.
    num_microbatches: See `dense_layer_computation()` in `dense.py`.

  Returns:
    See `dense_layer_computation()` in `dense.py`.
  """
  if input_kwargs:
    raise ValueError("Embedding layer calls should not receive kwargs.")
  del input_kwargs
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  input_ids = tf.cast(*input_args, tf.int32)
  base_vars = layer_instance(input_ids)
  tape.watch(base_vars)

  def sqr_norm_fn(grads):
    broadcast_axes = list(range(len(grads.shape)))
    del broadcast_axes[layer_instance._seq_axis]  # pylint: disable=protected-access
    del broadcast_axes[-1], broadcast_axes[0]
    reduced_grads = tf.reduce_sum(grads, axis=broadcast_axes)
    if num_microbatches is not None:
      reduced_grads = common_manip_utils.maybe_add_microbatch_axis(
          reduced_grads,
          num_microbatches,
      )
      reduced_grads = tf.reduce_sum(reduced_grads, axis=1)
    reduction_axes = tf.range(1, tf.rank(reduced_grads))
    return tf.reduce_sum(tf.square(reduced_grads), axis=reduction_axes)

  return base_vars, base_vars, sqr_norm_fn

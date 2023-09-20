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
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import registry_function_utils


def nlp_on_device_embedding_layer_computation(
    layer_instance: tf.keras.layers.Layer,
    input_args: Sequence[Any],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tfm.nlp.layers.OnDeviceEmbedding`.

  Args:
    layer_instance: A `tfm.nlp.layers.OnDeviceEmbedding` instance.
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
  if hasattr(layer_instance, "_use_one_hot"):
    if layer_instance._use_one_hot:  # pylint: disable=protected-access
      raise NotImplementedError(
          "The embedding feature '_use_one_hot' is not supported."
      )
  # NOTE: Since the implementation of `tfm.nlp.layers.OnDeviceEmbedding` uses
  # `.set_shape()`, we can assume that inputs are not ragged.
  input_ids = tf.cast(*input_args, tf.int32)
  if len(layer_instance.trainable_variables) != 1:
    raise ValueError(
        "Only layer instances with only one set of trainable variables"
        "are permitted."
    )
  base_vars = layer_instance.trainable_variables[0]
  tape.watch(base_vars)
  outputs = layer_instance(input_ids)

  def sqr_norm_fn(base_vars_grads: tf.IndexedSlices):
    return registry_function_utils.embedding_sqr_norm_fn(
        base_vars_grads.values,
        input_ids,
        num_microbatches,
    )

  return base_vars, outputs, sqr_norm_fn

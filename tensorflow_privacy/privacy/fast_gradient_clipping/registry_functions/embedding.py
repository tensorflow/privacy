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
"""Fast clipping function for `tf.keras.layers.Embedding`."""

from typing import Any, Dict, Optional, Text, Tuple
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def embedding_layer_computation(
    layer_instance: tf.keras.layers.Embedding,
    input_args: Tuple[Any, ...],
    input_kwargs: Dict[Text, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.Embedding`.

  The logic of this computation is based on the `tf.keras.layers.Dense`
  computation and the fact that an embedding layer is just a dense layer
  with no activation function and an output vector of the form X*W for input
  X, where the i-th row of W is the i-th embedding vector and the j-th row of
  X is a one-hot vector representing the input of example j.

  Args:
    layer_instance: A `tf.keras.layers.Embedding` instance.
    input_args: See `dense_layer_computation()` in `dense.py`.
    input_kwargs: See `dense_layer_computation()` in `dense.py`.
    tape: See `dense_layer_computation()` in `dense.py`.
    num_microbatches: See `dense_layer_computation()` in `dense.py`.

  Returns:
    See `dense_layer_computation()` in `dense.py`.
  """
  if input_kwargs:
    raise ValueError("Embedding layer calls should not receive kwargs.")
  del input_kwargs  # Unused in embedding layer calls.
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  if hasattr(layer_instance, "sparse"):  # for backwards compatibility
    if layer_instance.sparse:
      raise NotImplementedError("Sparse output tensors are not supported.")
  if isinstance(input_args[0], tf.SparseTensor):
    raise NotImplementedError("Sparse input tensors are not supported.")

  # Disable experimental features.
  if hasattr(layer_instance, "_use_one_hot_matmul"):
    if layer_instance._use_one_hot_matmul:  # pylint: disable=protected-access
      raise NotImplementedError(
          "The experimental embedding feature"
          "'_use_one_hot_matmul' is not supported."
      )
  input_ids = tf.cast(*input_args, tf.int32)
  base_vars = layer_instance.trainable_variables[0]
  tape.watch(base_vars)
  outputs = tf.nn.embedding_lookup(base_vars, input_ids)

  def sqr_norm_fn(base_vars_grads):
    # Get a 1D tensor of the row indices.
    nrows = tf.shape(input_ids)[0]
    if isinstance(input_ids, tf.RaggedTensor):
      row_indices = tf.expand_dims(
          input_ids.merge_dims(1, -1).value_rowids(), axis=-1
      )
    elif isinstance(input_ids, tf.Tensor):
      ncols = tf.reduce_prod(tf.shape(input_ids)[1:])
      repeats = tf.repeat(ncols, nrows)
      row_indices = tf.reshape(tf.repeat(tf.range(nrows), repeats), [-1, 1])
    else:
      raise NotImplementedError(
          "Cannot parse input_ids of type %s" % input_ids.__class__.__name__
      )
    row_indices = tf.cast(row_indices, tf.int32)
    if num_microbatches is not None:
      microbatch_size = tf.cast(nrows / num_microbatches, tf.int32)
      nrows = num_microbatches
      row_indices = tf.cast(
          tf.math.floordiv(row_indices, microbatch_size), tf.int32
      )
    # Sum-reduce the `IndexSlices` that is the result of a `tape.gradient()`
    # call. The sum is reduced by the repeated embedding indices and batch
    # index. It is adapted from the logic in:
    #   tf.keras.optimizers.legacy.optimizer_v2._deduplicate_indexed_slices
    if not isinstance(base_vars_grads, tf.IndexedSlices):
      raise NotImplementedError(
          "Cannot parse embedding gradients of type: %s"
          % base_vars_grads.__class__.__name__
      )
    slice_indices = tf.expand_dims(base_vars_grads.indices, axis=-1)
    paired_indices = tf.concat(
        [tf.cast(row_indices, tf.int64), tf.cast(slice_indices, tf.int64)],
        axis=1,
    )
    (unique_paired_indices, new_index_positions) = tf.raw_ops.UniqueV2(
        x=paired_indices, axis=[0]
    )
    unique_batch_ids = unique_paired_indices[:, 0]
    summed_gradients = tf.math.unsorted_segment_sum(
        base_vars_grads.values,
        new_index_positions,
        tf.shape(unique_paired_indices)[0],
    )
    # Compute the squared gradient norms at the per-example level.
    sqr_gradient_sum = tf.reduce_sum(tf.square(summed_gradients), axis=1)
    summed_data_range = tf.range(tf.shape(sqr_gradient_sum)[0])
    return tf.sparse.segment_sum(
        sqr_gradient_sum,
        summed_data_range,
        tf.sort(unique_batch_ids),
        num_segments=nrows,
    )  # fill in empty inputs

  return base_vars, outputs, sqr_norm_fn

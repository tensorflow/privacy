# Copyright 2024, The TensorFlow Authors.
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
"""Compute the contribution histogram for an embedding layer."""

from typing import Optional
import tensorflow as tf
from tensorflow_privacy.privacy.sparsity_preserving_noise import type_aliases


def embedding_layer_contribution_histogram(
    layer_instance: tf.keras.layers.Embedding,
    input_args: type_aliases.InputArgs,
    input_kwargs: type_aliases.InputKwargs,
    num_microbatches: Optional[tf.Tensor] = None,
) -> dict[str, type_aliases.ContributionCountHistogramFn]:
  """Registry function for `tf.keras.layers.Embedding`.

  Args:
    layer_instance: A `tf.keras.layers.Embedding` instance.
    input_args: A `tuple` containing the first part of `layer_instance` input.
      Specifically, `layer_instance(*inputs_args, **input_kwargs)` should return
      a valid output.
    input_kwargs: A `tuple` containing the second part of `layer_instance`
      input. Specifically, `layer_instance(*inputs_args, **input_kwargs)` should
      return a valid output.
    num_microbatches: An optional numeric value or scalar `tf.Tensor` for
      indicating whether and how the losses are grouped into microbatches. If
      not None, num_microbatches must divide the batch size.

  Returns:
    A dict mapping the name of the trainable variable to a function with
    signature `(tf.IndexedSlices) -> tf.SparseTensor`. The function takes a
    `tf.IndexedSlices` object representing the gradient for that variable and
    returns a `tf.SparseTensor` representing the normalized (so that each user
    contributes 1) contribution counts histogram per user for each embedding
    vector.
  """
  if input_kwargs:
    raise ValueError("Embedding layer calls should not receive kwargs.")
  del input_kwargs  # Unused in embedding layer calls.
  if not input_args or len(input_args) != 1:
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
          "The experimental embedding feature "
          "'_use_one_hot_matmul' is not supported."
      )
  input_ids = tf.squeeze(tf.cast(*input_args, tf.int32))

  def count_contributions_fn(
      grad: type_aliases.SparseGradient,
  ) -> type_aliases.ContributionCountHistogram:
    return embedding_layer_contribution_histogram_fn(
        grad,
        input_ids,
        layer_instance.input_dim,
        num_microbatches,
    )

  if (
      not layer_instance.trainable_variables
      or len(layer_instance.trainable_variables) != 1
  ):
    raise ValueError(
        "Embedding layer must have exactly one trainable variable."
    )
  return {layer_instance.trainable_variables[0].name: count_contributions_fn}


def embedding_layer_contribution_histogram_fn(
    grad: type_aliases.SparseGradient,
    input_ids: tf.Tensor,
    vocab_size: Optional[tf.Tensor],
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.ContributionCountHistogram:
  """Computes the normalized contribution counts histogram for embedding layer.

  NOTE: to help understand the code, we document in the function body what the
  expected intermediate variables are for the below running example:

      grad = None
      input_ids = [[1, 1, 2], [0], [2, 0]]
      vocab_size = 3
      num_microbatches = None

  For ease of reference, we also list these variables below:

    row_indices = [[0], [0], [0], [1], [2], [2]]
    flattened_indices = [[1], [1], [2], [0], [2], [0]]
    paired_indices = [[0, 1], [0, 1], [0, 2], [1, 0], [2, 2], [2, 0]]
    linearized_pair_indices = [1 1 2 3 8 6]
    contribution_counts_linearized_indices = [1 2 3 8 6]
    contribution_counts_indices = [[0, 1], [0, 2], [1, 0], [2, 2], [2, 0]]
    contribution_counts_values = [2 1 1 1 1]
    user_normalized_contribution_counts = tf.SparseTensor(
        indices=[[0, 1], [0, 2], [1, 0], [2, 0], [2, 2]],
        values=[0.67, 0.33, 1., 0.5, 0.5,]
        shape=(3, 3)
    )
    contribution_histogram = tf.SparseTensor(
        indices=[[0], [1], [2]],
        values=[1.5, 0.67, 0.83],
        shape=(3,)
    )


  Args:
    grad: The gradient of the layer. (unused for embedding layer)
    input_ids: The input ids used to compute the embeddings.
    vocab_size: The vocabulary size of the embedding layer.
    num_microbatches: An optional numeric value or scalar `tf.Tensor` for
      indicating whether and how the losses are grouped into microbatches. If
      not None, num_microbatches must divide the batch size.

  Returns:
    A `tf.SparseTensor` representing the normalized (so that each user
    contributes 1) contribution counts histogram per user for each embedding
    vector.

  Raises:
    NotImplementedError: If the input_ids is not a `tf.Tensor` or
      `tf.RaggedTensor`.
  """
  del grad  # unused.

  nrows = tf.shape(input_ids)[0]
  if isinstance(input_ids, tf.RaggedTensor):
    row_indices = tf.expand_dims(
        input_ids.merge_dims(1, -1).value_rowids(), axis=-1
    )
  elif isinstance(input_ids, tf.Tensor):
    ncols = tf.reduce_prod(tf.shape(input_ids)[1:])
    repeats = tf.repeat(ncols, nrows)
    row_indices = tf.reshape(tf.repeat(tf.range(nrows), repeats), [-1, 1])
    row_indices = tf.cast(row_indices, tf.int64)
  else:
    raise NotImplementedError(
        "Cannot parse input_ids of type %s" % input_ids.__class__.__name__
    )

  if num_microbatches is not None:
    tf.debugging.assert_equal(
        nrows % num_microbatches,
        0,
        "num_microbatches must divide the batch size.",
    )
    microbatch_size = tf.cast(nrows / num_microbatches, tf.int64)
    nrows = num_microbatches
    row_indices = tf.cast(
        tf.math.floordiv(row_indices, microbatch_size), tf.int64
    )
  # NOTE: expected values for the running example above are
  #   row_indices = [[0], [0], [0], [1], [2], [2]]

  flattened_indices = tf.expand_dims(tf.reshape(input_ids, [-1]), axis=-1)
  paired_indices = tf.concat(
      [tf.cast(row_indices, tf.int64), tf.cast(flattened_indices, tf.int64)],
      axis=1,
  )
  # NOTE: expected values for the running example above are
  #   flattened_indices = [[1], [1], [2], [0], [2], [0]]
  #   paired_indices = [[0, 1], [0, 1], [0, 2], [1, 0], [2, 2], [2, 0]]

  transform = tf.cast(tf.stack([[vocab_size], [1]], axis=0), tf.int64)
  linearized_pair_indices = tf.reshape(
      tf.matmul(paired_indices, transform), (-1,)
  )
  contribution_counts_linearized_indices, _, contribution_counts_values = (
      tf.unique_with_counts(linearized_pair_indices)
  )
  contribution_counts_indices = tf.stack(
      [
          contribution_counts_linearized_indices // vocab_size,
          contribution_counts_linearized_indices % vocab_size,
      ],
      axis=1,
  )
  contribution_counts = tf.sparse.SparseTensor(
      contribution_counts_indices,
      contribution_counts_values,
      (nrows, vocab_size),
  )
  contribution_counts = tf.sparse.reorder(contribution_counts)
  # NOTE: expected values for the running example above are
  #   linearized_pair_indices = [1 1 2 3 8 6]
  #   contribution_counts_linearized_indices = [1 2 3 8 6]
  #   contribution_counts_indices = [[0, 1], [0, 2], [1, 0], [2, 2], [2, 0]]
  #   contribution_counts_values = [2 1 1 1 1]

  user_normalized_contribution_counts = (
      contribution_counts
      / tf.sparse.reduce_sum(contribution_counts, axis=-1, keepdims=True)
  )
  contribution_histogram = tf.sparse.reduce_sum(
      user_normalized_contribution_counts, axis=0, output_is_sparse=True
  )
  # NOTE: expected values for the running example above are
  #   user_normalized_contribution_counts = tf.SparseTensor(
  #       indices=[[0, 1], [0, 2], [1, 0], [2, 0], [2, 2]],
  #       values=[0.67, 0.33, 1., 0.5, 0.5,]
  #       shape=(3, 3)
  #   )
  #   contribution_histogram = tf.SparseTensor(
  #       indices=[[0], [1], [2]],
  #       values=[1.5, 0.67, 0.83],
  #       shape=(3,)
  #   )

  return tf.sparse.reshape(contribution_histogram, (-1,))

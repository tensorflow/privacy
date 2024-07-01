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
"""Utils for adding sparse noise to gradients."""

from typing import Mapping, Optional

from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp


def split_noise_multiplier(
    noise_multiplier: float,
    sparse_selection_ratio: float,
    sparse_selection_contribution_counts: Optional[list[tf.SparseTensor]],
) -> tuple[float, float]:
  """Split noise multiplier between partition selection and gradient noise.

  Args:
    noise_multiplier: The original noise multiplier.
    sparse_selection_ratio: The ratio of partition selection noise and gradient
      noise.
    sparse_selection_contribution_counts: The contribution counts for each
      sparse selection variable.

  Returns:
    A tuple of noise multipliers for sparse selection and gradient noise.
  """
  assert (
      sparse_selection_ratio > 0.0 and sparse_selection_ratio < 1.0
  ), 'Sparse selection ratio must be between 0 and 1.'
  assert (
      sparse_selection_contribution_counts is not None
  ), 'Sparse selection contribution counts must not be None.'
  num_sparse_selections = len(
      [c for c in sparse_selection_contribution_counts if c is not None]
  )
  assert (
      num_sparse_selections > 0
  ), 'No sparse selections contribution counts found.'

  ratio = (1 + sparse_selection_ratio**2.0) ** 0.5
  total_noise_multiplier_sparse = noise_multiplier * ratio
  noise_multiplier_sparse = (
      num_sparse_selections * total_noise_multiplier_sparse**2.0
  ) ** 0.5
  noise_multiplier_dense = noise_multiplier * ratio / sparse_selection_ratio

  return noise_multiplier_sparse, noise_multiplier_dense


def _sample_sparse_indices_batch_size_heuristic(
    max_index: tf.Tensor, prob: float
) -> int:
  """A rough heuristic for the batch size to use for sampling.

  This heuristic should roughly allow for the sampling to only use a single
  batch to sample all indices >95% of the time.

  Args:
    max_index: The maximum index to sample.
    prob: The probability of sampling each index.

  Returns:
    The batch size to use for sampling.
  """
  max_num_samples = tf.cast(max_index + 1, tf.float32)
  expected_num_samples = max_num_samples * prob
  return tf.cast(tf.maximum(50.0, 1.2 * expected_num_samples), tf.int32)


@tf.function
def sample_sparse_indices(
    max_index: tf.Tensor, prob: float, batch_size: int | None = None
) -> tf.Tensor:
  """Samples indices with probability `prob` iid sparsely.

  This function generates a list of indices in the range of [0, max_index]
  where each index is sampled with probability `prob` independently. To achieve
  this efficiently, we use the geometric distribution to sample a batch of
  indices at a time and repeat this process until all indices are sampled.

  Args:
    max_index: The maximum index to sample.
    prob: The probability of sampling each index.
    batch_size: (optional) The batch size to use for sampling. If not provided,
      a heuristic will be used to determine the batch size.

  Returns:
    A tensor of sampled indices.
  """
  if prob <= 0.0:
    return tf.constant([], dtype=tf.int64)

  ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

  batch_size = batch_size or _sample_sparse_indices_batch_size_heuristic(
      max_index, prob
  )

  geom = tfp.distributions.geometric.Geometric(probs=prob)

  i, current_max = tf.constant(0), tf.constant(0)
  while current_max <= max_index:
    sample = tf.cast(geom.sample(batch_size) + 1, tf.int32)
    indices = current_max + tf.cumsum(sample)
    current_max = tf.reduce_max(indices)
    ta = ta.write(i, indices)
    i += 1

  indices = tf.cast(ta.concat(), tf.int32) - 1
  indices = indices[indices <= max_index]
  return tf.cast(indices, tf.int64)


def sample_indices(
    contribution_counts: tf.SparseTensor,
    noise_multiplier: float,
    threshold: int,
) -> tf.Tensor:
  """Samples indices where the count + gaussian noise is above a threshold.

  Args:
    contribution_counts: The contribution counts for each index.
    noise_multiplier: The noise multiplier to use for the gaussian noise.
    threshold: The threshold to use for the selection.

  Returns:
    A tensor of sampled indices.
  """
  contribution_count_values = tf.reshape(contribution_counts.values, (-1,))
  noised_contribution_count_values = (
      contribution_count_values
      + tf.random.normal(
          tf.shape(contribution_count_values),
          mean=0.0,
          stddev=noise_multiplier,
          dtype=tf.float32,
      )
  )
  noised_contribution_counts_indices = contribution_counts.indices[
      noised_contribution_count_values >= threshold
  ][:, 0]
  return tf.reshape(noised_contribution_counts_indices, (-1,))


def sparse_private_partition_selection(
    contribution_counts: tf.SparseTensor,
    noise_multiplier: float,
    threshold: int,
) -> tf.Tensor:
  """Differentially private partition selection.

  Uses the sparse sampling algorithm to sample false positive indices. Also
  assumes that the contribution counts are clipped to a per example contribution
  of 1.

  Args:
    contribution_counts: The contribution counts for each index.
    noise_multiplier: The noise multiplier to use for the gaussian noise.
    threshold: The threshold to use for the selection.

  Returns:
    A tensor of selected indices.
  """
  if threshold < 0:
    raise ValueError(f'Threshold must be positive, got {threshold}.')

  true_positive_indices = sample_indices(
      contribution_counts, noise_multiplier, threshold
  )

  if noise_multiplier <= 0.0:
    return true_positive_indices

  # probability of selecting an index with zero contribution count.
  prob = stats.norm.sf(threshold / noise_multiplier).item()

  num_total_indices = tf.cast(contribution_counts.dense_shape[0], tf.int32)
  num_non_zero_indices = tf.shape(contribution_counts.values)[0]
  max_index = tf.cast(num_total_indices - num_non_zero_indices - 1, tf.int32)
  false_positive_indices = sample_sparse_indices(max_index, prob)

  below_counts = tf.searchsorted(
      true_positive_indices, false_positive_indices, side='right'
  )
  false_positives_fixed = false_positive_indices + tf.cast(
      below_counts, tf.int64
  )

  merged_indices = tf.concat(
      [false_positives_fixed, true_positive_indices], axis=0
  )
  return merged_indices


@tf.function
def _add_indexed_slices(*xs: tf.IndexedSlices) -> tf.IndexedSlices:
  """Adds a list of `tf.IndexedSlices` of the same shape."""
  output_dense_shape = tf.shape(xs[0])
  for x in xs[1:]:
    tf.debugging.assert_equal(output_dense_shape, tf.shape(x))

  indices = tf.concat([tf.cast(x.indices, tf.int64) for x in xs], axis=0)
  values = tf.concat([x.values for x in xs], axis=0)
  unique_indices, new_index_positions = tf.unique(indices)
  summed_values = tf.math.unsorted_segment_sum(
      values, new_index_positions, tf.shape(unique_indices)[0]
  )
  return tf.IndexedSlices(
      values=summed_values,
      indices=unique_indices,
      dense_shape=output_dense_shape,
  )


def add_sparse_gradient_noise(
    grad: tf.IndexedSlices, indices: tf.Tensor, noise_stddev: float
) -> tf.IndexedSlices:
  """Adds sparse gradient noise.

  Args:
    grad: A sparse gradient of type `tf.IndexedSlices`.
    indices: The selected indices to keep.
    noise_stddev: The standard deviation of the noise to add.

  Returns:
    A sparse gradient of type `tf.IndexedSlices` with the noise added.
  """
  noise_shape = tf.concat(
      [tf.shape(indices)[:1], tf.shape(grad.values)[1:]], axis=0
  )
  sparse_noise_values = tf.random.normal(
      noise_shape, mean=0.0, stddev=noise_stddev
  )
  sparse_noise = tf.IndexedSlices(
      indices=indices, values=sparse_noise_values, dense_shape=grad.dense_shape
  )
  return _add_indexed_slices(grad, sparse_noise)


def get_contribution_counts(
    trainable_vars: list[tf.Variable],
    grads: list[tf.Tensor],
    varname_to_contribution_counts_fns: Mapping[str, tf.SparseTensor],
) -> list[tf.Tensor | None]:
  """Gets the contribution counts for each variable in the Model."""
  contribution_counts_list = []
  for var, grad in zip(trainable_vars, grads):
    varname = var.name
    if varname not in varname_to_contribution_counts_fns:
      contribution_counts_list.append(None)
      continue
    contribution_counts_fns = varname_to_contribution_counts_fns[varname]
    if not contribution_counts_fns or not contribution_counts_fns[0]:
      contribution_counts_list.append(None)
      continue
    if len(contribution_counts_fns) > 1:
      raise NotImplementedError(
          'Sparse noise is not supported for shared weight variables.'
      )
    contribution_counts_fn = contribution_counts_fns[0]
    contribution_counts = contribution_counts_fn(grad)
    contribution_counts_list.append(contribution_counts)

  return contribution_counts_list


def add_sparse_noise(
    grad: tf.IndexedSlices,
    contribution_counts: tf.SparseTensor,
    noise_multiplier: float,
    noise_multiplier_sparse: float,
    l2_norm_clip: float,
    threshold: int,
) -> tf.IndexedSlices:
  """Adds sparse noise to a gradient.

  Args:
    grad: A sparse gradient of type `tf.IndexedSlices`.
    contribution_counts: The contribution counts for each index of grad.
    noise_multiplier: The noise multiplier to use for the gradient noise.
    noise_multiplier_sparse: The noise multiplier to use for the partition
      selection.
    l2_norm_clip: The l2 norm clip at which the gradient is clipped.
    threshold: The threshold to use for the partition selection.

  Returns:
    A sparse gradient of type `tf.IndexedSlices` with the noise added.
  """
  privately_selected_indices = sparse_private_partition_selection(
      contribution_counts, noise_multiplier_sparse, threshold
  )
  noised_grad = add_sparse_gradient_noise(
      grad, privately_selected_indices, noise_multiplier * l2_norm_clip
  )
  return noised_grad

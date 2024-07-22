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
"""Utils for adding sparse noise to gradients.

For more details on the algorithm, refer to https://arxiv.org/abs/2311.08357.
"""

from typing import Optional, Sequence

import tensorflow as tf
import tensorflow_probability as tfp


def split_noise_multiplier(
    noise_multiplier: float,
    sparse_selection_ratio: float,
    sparse_selection_contribution_counts: Sequence[Optional[tf.SparseTensor]],
) -> tuple[float, float]:
  """Splits noise multiplier between partition selection and gradient noise.

  Returns one noise multiplier for gradient noise and one noise multiplier
  for each sparse partition selection layer such that composing all gaussian
  mechanisms with these noise multipliers is equivalent to applying a single
  gaussian mechanism with the original noise multiplier.

  Args:
    noise_multiplier: The original noise multiplier.
    sparse_selection_ratio: The ratio of partition selection noise and gradient
      noise.
    sparse_selection_contribution_counts: The contribution counts for each
      sparse selection variable. If a sparse selection count is None, it will be
      ignored.

  Returns:
    A tuple of noise multipliers for sparse selection and gradient noise.

  Raises:
    ValueError: If the sparse selection ratio is not between 0 and 1, if the
      sparse selection contribution counts is None, or if there are no sparse
      selection contribution counts.
  """
  if sparse_selection_ratio <= 0.0 or sparse_selection_ratio >= 1.0:
    raise ValueError('Sparse selection ratio must be between 0 and 1.')
  num_sparse_selections = sum(
      1 for c in sparse_selection_contribution_counts if c is not None
  )
  if num_sparse_selections == 0:
    raise ValueError('No sparse selections contribution counts found.')

  ratio = (1.0 + sparse_selection_ratio**2.0) ** 0.5
  total_noise_multiplier_sparse = noise_multiplier * ratio
  noise_multiplier_partition_selection = (
      num_sparse_selections**0.5 * total_noise_multiplier_sparse
  )
  noise_multiplier_gradient_noise = (
      noise_multiplier * ratio / sparse_selection_ratio
  )

  return noise_multiplier_partition_selection, noise_multiplier_gradient_noise


def _sample_sparse_indices_batch_size_heuristic(
    max_index: tf.Tensor,
    probability: float,
) -> tf.Tensor:
  """Returns a batch size using a rough heuristic to use for sampling.

  This heuristic should roughly allow for the sampling to only use a single
  batch to sample all indices >95% of the time.

  Args:
    max_index: The maximum index to sample.
    probability: The probability of sampling each index.

  Returns:
    The batch size to use for sampling.
  """
  max_num_samples = tf.cast(max_index + 1, tf.float32)
  expected_num_samples = max_num_samples * probability
  # For expected samples > 50, choosing a batch size of 1.2 * expected samples
  # will allow for sampling only once to get all indices >95% of the time.
  min_batch_size = 50.0
  return tf.cast(
      tf.maximum(min_batch_size, 1.2 * expected_num_samples), tf.int32
  )


@tf.function
def sample_false_positive_indices(
    max_index: tf.Tensor, probability: float, batch_size: Optional[int] = None
) -> tf.Tensor:
  """Samples indices with probability `probability` iid sparsely.

  This function generates a list of indices in the range of [0, max_index]
  where each index is sampled with probability `probability` independently. To
  achieve this efficiently, we use the geometric distribution to sample a batch
  of indices at a time and repeat this process until all indices are sampled.

  Args:
    max_index: The maximum index to sample.
    probability: The probability of sampling each index.
    batch_size: The batch size to use for sampling. If None, a heuristic will be
      used to determine the batch size.

  Returns:
    A tensor of sampled indices.
  """
  if probability <= 0.0:
    return tf.constant([], dtype=tf.int64)

  sampled_indices = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

  batch_size = batch_size or _sample_sparse_indices_batch_size_heuristic(
      max_index, probability
  )

  geom = tfp.distributions.geometric.Geometric(probs=probability)

  i, current_max = tf.constant(0), tf.constant(-1)
  while current_max < max_index:
    sample = tf.cast(geom.sample(batch_size) + 1, tf.int32)
    indices = current_max + tf.cumsum(sample)
    current_max = indices[-1]
    sampled_indices = sampled_indices.write(i, indices)
    i += 1

  indices = tf.cast(sampled_indices.concat(), tf.int32)
  indices = indices[indices <= max_index]
  return tf.cast(indices, tf.int64)


def sample_true_positive_indices(
    contribution_counts: tf.SparseTensor,
    noise_multiplier: float,
    threshold: int,
) -> tf.Tensor:
  """Samples indices where the count + Gaussian noise is above a threshold.

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

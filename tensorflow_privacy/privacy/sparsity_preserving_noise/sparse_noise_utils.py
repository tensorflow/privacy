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

import collections
from typing import Mapping, Optional, Sequence

from scipy import stats
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.sparsity_preserving_noise import type_aliases
import tensorflow_probability as tfp


def split_noise_multiplier(
    noise_multiplier: float,
    sparse_selection_privacy_budget_fraction: float,
    sparse_selection_contribution_counts: Sequence[Optional[tf.SparseTensor]],
) -> tuple[float, float]:
  """Splits noise multiplier between partition selection and gradient noise.

  Returns one noise multiplier for gradient noise and one noise multiplier
  for each sparse partition selection layer such that composing all gaussian
  mechanisms with these noise multipliers is equivalent to applying a single
  gaussian mechanism with the original noise multiplier.

  Args:
    noise_multiplier: The original noise multiplier.
    sparse_selection_privacy_budget_fraction: The fraction of privacy budget to
      use for partition selection.
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
  if (
      sparse_selection_privacy_budget_fraction <= 0.0
      or sparse_selection_privacy_budget_fraction >= 1.0
  ):
    raise ValueError(
        'Sparse selection privacy budget fraction must be between 0 and 1.'
    )
  num_sparse_selections = sum(
      1 for c in sparse_selection_contribution_counts if c is not None
  )
  if num_sparse_selections == 0:
    raise ValueError('No sparse selections contribution counts found.')

  sparse_selection_ratio = sparse_selection_privacy_budget_fraction / (
      1.0 - sparse_selection_privacy_budget_fraction
  )
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
          dtype=contribution_count_values.dtype,
      )
  )
  noised_contribution_counts_indices = contribution_counts.indices[
      noised_contribution_count_values >= threshold
  ][:, 0]
  return tf.reshape(noised_contribution_counts_indices, (-1,))


@tf.function
def _remap_indices(
    indices: tf.Tensor,
    skip_indices: tf.Tensor,
) -> tf.Tensor:
  """Remaps the indices while skipping the skip indices.

  As an example, if skip_indices = [1, 3], then the indices will be remapped as
  follows:
  0 -> 0
  1 -> 2
  2 -> 4
  3 -> 5
  4 -> 6
  5 -> 7
  ...

  This is useful for merging the true positive and false positive indices.

  Args:
    indices: The indices to remap.
    skip_indices: The indices to skip while remapping. Assumed to be sorted.

  Returns:
    The remapped indices.
  """

  def piecewise_map(skip_indices):
    map_counts = tf.range(tf.size(skip_indices) + 1, dtype=tf.int64)
    map_idx = skip_indices - map_counts[:-1]
    skip_indices = tf.concat([[-1], skip_indices], axis=0)
    gaps = skip_indices[1:] - skip_indices[:-1] - 1

    map_idx = tf.concat([map_idx, [tf.int64.max]], axis=0)
    gaps = tf.concat([gaps, [1]], axis=0)

    return map_idx[gaps > 0], map_counts[gaps > 0]

  map_idx, map_count = piecewise_map(skip_indices)
  idx = tf.searchsorted(map_idx, indices, side='right')
  offset = tf.gather(map_count, idx)
  return indices + offset


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

  true_positive_indices = sample_true_positive_indices(
      contribution_counts, noise_multiplier, threshold
  )

  if noise_multiplier <= 0.0:
    return true_positive_indices

  # probability of selecting an index with zero contribution count.
  prob = stats.norm.sf(threshold / noise_multiplier).item()

  num_total_indices = tf.cast(contribution_counts.dense_shape[0], tf.int32)
  num_non_zero_indices = tf.shape(contribution_counts.values)[0]
  max_index = tf.cast(num_total_indices - num_non_zero_indices - 1, tf.int32)
  false_positive_indices = sample_false_positive_indices(max_index, prob)
  remapped_false_positive_indices = _remap_indices(
      false_positive_indices, tf.reshape(contribution_counts.indices, (-1,))
  )
  merged_indices = tf.sort(
      tf.concat(
          [remapped_false_positive_indices, true_positive_indices], axis=0
      )
  )
  return merged_indices


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
  filtered_grad_values = tf.gather(grad, indices)
  sparse_noise_values = tf.random.normal(
      tf.shape(filtered_grad_values), mean=0.0, stddev=noise_stddev
  )
  filtered_noised_grad_values = filtered_grad_values + sparse_noise_values
  return tf.IndexedSlices(
      indices=indices,
      values=filtered_noised_grad_values,
      dense_shape=grad.dense_shape,
  )


def extract_varname_to_contribution_counts_fns(
    registry_fn_outputs_list: Sequence[
        gradient_clipping_utils.RegistryGeneratorFunctionOutput
    ],
    trainable_vars: Sequence[tf.Variable],
) -> Mapping[str, type_aliases.ContributionCountHistogramFn]:
  """Extracts a map of contribution count fns from generator outputs.

  Args:
    registry_fn_outputs_list: A list of `RegistryGeneratorFunctionOutput`
      instances returned by
      `gradient_clipping_utils.model_forward_backward_pass`.
    trainable_vars: A list of trainable variables.

  Returns:
    A `dict` from varname to contribution counts functions
  """
  if trainable_vars is not None:
    # Create a set using `ref()` for fast set membership check. tf.Variable
    # itself is not hashable.
    trainable_vars = set([v.ref() for v in trainable_vars])

  varname_to_contribution_counts_fns = collections.defaultdict(list)
  for registry_fn_output in registry_fn_outputs_list:
    if trainable_vars is None or any(
        w.ref() in trainable_vars
        for w in registry_fn_output.layer_trainable_weights
    ):
      if registry_fn_output.varname_to_count_contribution_fn is not None:
        duplicate_varnames = set(
            registry_fn_output.varname_to_count_contribution_fn.keys()
        ) & set(varname_to_contribution_counts_fns.keys())
        if duplicate_varnames:
          raise ValueError(
              'Duplicate varnames: {duplicate_varnames} found in contribution'
              ' counts functions.'
          )
        varname_to_contribution_counts_fns.update(
            registry_fn_output.varname_to_count_contribution_fn
        )
  return varname_to_contribution_counts_fns


def get_contribution_counts(
    trainable_vars: Sequence[tf.Variable],
    grads: Sequence[tf.Tensor],
    varname_to_contribution_counts_fns: Mapping[
        str, type_aliases.ContributionCountHistogramFn
    ],
) -> Sequence[type_aliases.ContributionCountHistogram | None]:
  """Gets the contribution counts for each variable in the Model.

  Args:
    trainable_vars: A list of trainable variables.
    grads: A corresponding list of gradients for each trainable variable.
    varname_to_contribution_counts_fns: A mapping from variable name to a list
      of functions to get the contribution counts for that variable.

  Returns:
    A list of contribution counts for each variable and None for variables that
    do not have contribution counts function.

  Raises:
    NotImplementedError: If there are more than one contribution counts function
      for a variable.
  """
  contribution_counts_list = []
  for var, grad in zip(trainable_vars, grads):
    if var.name not in varname_to_contribution_counts_fns:
      contribution_counts_list.append(None)
      continue
    contribution_counts_fn = varname_to_contribution_counts_fns[var.name]
    if not contribution_counts_fn:
      contribution_counts_list.append(None)
      continue
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

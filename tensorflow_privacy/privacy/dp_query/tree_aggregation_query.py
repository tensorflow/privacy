# Copyright 2021, The TensorFlow Authors.
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
"""DPQuery for continual observation queries relying on `tree_aggregation`."""

import attr
import tensorflow as tf

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation


class TreeCumulativeSumQuery(dp_query.SumAggregationDPQuery):
  """Implements dp_query for adding correlated noise through tree structure.

  First clips and sums records in current sample, returns cumulative sum of
  samples over time (instead of only current sample) with added noise for
  cumulative sum proportional to log(T), T being the number of times the query
  is called.

  Attributes:
    clip_fn: Callable that specifies clipping function. `clip_fn` receives two
      arguments: a flat list of vars in a record and a `clip_value` to clip the
      corresponding record, e.g. clip_fn(flat_record, clip_value).
    clip_value: float indicating the value at which to clip the record.
    record_specs: `Collection[tf.TensorSpec]` specifying shapes of records.
    tree_aggregator: `tree_aggregation.TreeAggregator` initialized with
      user defined `noise_generator`. `noise_generator` is a
      `tree_aggregation.ValueGenerator` to generate the noise value for a tree
      node. Noise stdandard deviation is specified outside the `dp_query` by the
      user when defining `noise_fn` and should have order
      O(clip_norm*log(T)/eps) to guarantee eps-DP.
  """

  @attr.s(frozen=True)
  class GlobalState(object):
    """Class defining global state for Tree sum queries.

    Attributes:
      tree_state: Current state of noise tree keeping track of current leaf and
        each level state.
      clip_value: The clipping value to be passed to clip_fn.
      samples_cumulative_sum: Noiseless cumulative sum of samples over time.
    """
    tree_state = attr.ib()
    clip_value = attr.ib()
    samples_cumulative_sum = attr.ib()

  def __init__(self,
               record_specs,
               noise_generator,
               clip_fn,
               clip_value,
               use_efficient=True):
    """Initializes the `TreeCumulativeSumQuery`.

    Consider using `build_l2_gaussian_query` for the construction of a
    `TreeCumulativeSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_generator: `tree_aggregation.ValueGenerator` to generate the noise
        value for a tree node. Should be coupled with clipping norm to guarantee
        privacy.
      clip_fn: Callable that specifies clipping function. Input to clip is a
        flat list of vars in a record.
      clip_value: Float indicating the value at which to clip the record.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    self._clip_fn = clip_fn
    self._clip_value = clip_value
    self._record_specs = record_specs
    if use_efficient:
      self._tree_aggregator = tree_aggregation.EfficientTreeAggregator(
          noise_generator)
    else:
      self._tree_aggregator = tree_aggregation.TreeAggregator(noise_generator)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    initial_tree_state = self._tree_aggregator.init_state()
    initial_samples_cumulative_sum = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape), self._record_specs)
    initial_state = TreeCumulativeSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        samples_cumulative_sum=initial_samples_cumulative_sum)
    return initial_state

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.clip_value

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Args:
      params: `clip_value` for the record.
      record: The record to be processed.

    Returns:
      Structure of clipped tensors.
    """
    clip_value = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list = self._clip_fn(record_as_list, clip_value)
    return tf.nest.pack_sequence_as(record, clipped_as_list)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    Updates tree state, and returns noised cumulative sum and updated state.

    Computes new cumulative sum, and returns its noised value. Grows tree state
    by one new leaf, and returns the new state.

    Args:
      sample_state: Sum of clipped records for this round.
      global_state: Global state with current sample's cumulative sum and tree
        state.

    Returns:
      A tuple of (noised_cumulative_sum, new_global_state).
    """
    new_cumulative_sum = tf.nest.map_structure(
        tf.add, global_state.samples_cumulative_sum, sample_state)
    cumulative_sum_noise, new_tree_state = self._tree_aggregator.get_cumsum_and_update(
        global_state.tree_state)
    new_global_state = attr.evolve(
        global_state,
        samples_cumulative_sum=new_cumulative_sum,
        tree_state=new_tree_state)
    noised_cum_sum = tf.nest.map_structure(tf.add, new_cumulative_sum,
                                           cumulative_sum_noise)
    return noised_cum_sum, new_global_state

  @classmethod
  def build_l2_gaussian_query(cls,
                              clip_norm,
                              noise_multiplier,
                              record_specs,
                              noise_seed=None,
                              use_efficient=True):
    """Returns a query instance with L2 norm clipping and Gaussian noise.

    Args:
      clip_norm: Each record will be clipped so that it has L2 norm at most
        `clip_norm`.
      noise_multiplier: The effective noise multiplier for the sum of records.
        Noise standard deviation is `clip_norm*noise_multiplier`.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_seed: Integer seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    if clip_norm <= 0:
      raise ValueError(f'`clip_norm` must be positive, got {clip_norm}.')

    if noise_multiplier < 0:
      raise ValueError(
          f'`noise_multiplier` must be non-negative, got {noise_multiplier}.')

    gaussian_noise_generator = tree_aggregation.GaussianNoiseGenerator(
        noise_std=clip_norm * noise_multiplier,
        specs=record_specs,
        seed=noise_seed)

    def l2_clip_fn(record_as_list, clip_norm):
      clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_norm)
      return clipped_record

    return cls(
        clip_fn=l2_clip_fn,
        clip_value=clip_norm,
        record_specs=record_specs,
        noise_generator=gaussian_noise_generator,
        use_efficient=use_efficient)


class TreeResidualSumQuery(dp_query.SumAggregationDPQuery):
  """Implements dp_query for adding correlated noise through tree structure.

  Clips and sums records in current sample; returns the current sample adding
  the noise residual from tree aggregation. The returned value is conceptually
  equivalent to the following: calculates cumulative sum of samples over time
  (instead of only current sample) with added noise for cumulative sum
  proportional to log(T), T being the number of times the query is called;
  returns the residual between the current noised cumsum and the previous one
  when the query is called. Combining this query with a SGD optimizer can be
  used to implement the DP-FTRL algorithm in
  "Practical and Private (Deep) Learning without Sampling or Shuffling".

  Attributes:
    clip_fn: Callable that specifies clipping function. `clip_fn` receives two
      arguments: a flat list of vars in a record and a `clip_value` to clip the
      corresponding record, e.g. clip_fn(flat_record, clip_value).
    clip_value: float indicating the value at which to clip the record.
    record_specs: A nested structure of `tf.TensorSpec`s specifying structure
      and shapes of records.
    tree_aggregator: `tree_aggregation.TreeAggregator` initialized with user
      defined `noise_generator`. `noise_generator` is a
      `tree_aggregation.ValueGenerator` to generate the noise value for a tree
      node. Noise stdandard deviation is specified outside the `dp_query` by the
      user when defining `noise_fn` and should have order
      O(clip_norm*log(T)/eps) to guarantee eps-DP.
  """

  @attr.s(frozen=True)
  class GlobalState(object):
    """Class defining global state for Tree sum queries.

    Attributes:
      tree_state: Current state of noise tree keeping track of current leaf and
        each level state.
      clip_value: The clipping value to be passed to clip_fn.
      previous_tree_noise: Cumulative noise by tree aggregation from the
        previous time the query is called on a sample.
    """
    tree_state = attr.ib()
    clip_value = attr.ib()
    previous_tree_noise = attr.ib()

  def __init__(self,
               record_specs,
               noise_generator,
               clip_fn,
               clip_value,
               use_efficient=True):
    """Initializes the `TreeResidualSumQuery`.

    Consider using `build_l2_gaussian_query` for the construction of a
    `TreeResidualSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_generator: `tree_aggregation.ValueGenerator` to generate the noise
        value for a tree node. Should be coupled with clipping norm to guarantee
        privacy.
      clip_fn: Callable that specifies clipping function. Input to clip is a
        flat list of vars in a record.
      clip_value: Float indicating the value at which to clip the record.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    self._clip_fn = clip_fn
    self._clip_value = clip_value
    self._record_specs = record_specs
    if use_efficient:
      self._tree_aggregator = tree_aggregation.EfficientTreeAggregator(
          noise_generator)
    else:
      self._tree_aggregator = tree_aggregation.TreeAggregator(noise_generator)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    initial_tree_state = self._tree_aggregator.init_state()
    initial_noise = tf.nest.map_structure(lambda spec: tf.zeros(spec.shape),
                                          self._record_specs)
    return TreeResidualSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        previous_tree_noise=initial_noise)

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.clip_value

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Args:
      params: `clip_value` for the record.
      record: The record to be processed.

    Returns:
      Structure of clipped tensors.
    """
    clip_value = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list = self._clip_fn(record_as_list, clip_value)
    return tf.nest.pack_sequence_as(record, clipped_as_list)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    Updates tree state, and returns residual of noised cumulative sum.

    Args:
      sample_state: Sum of clipped records for this round.
      global_state: Global state with current samples cumulative sum and tree
        state.

    Returns:
      A tuple of (noised_cumulative_sum, new_global_state).
    """
    tree_noise, new_tree_state = self._tree_aggregator.get_cumsum_and_update(
        global_state.tree_state)
    noised_sample = tf.nest.map_structure(lambda a, b, c: a + b - c,
                                          sample_state, tree_noise,
                                          global_state.previous_tree_noise)
    new_global_state = attr.evolve(
        global_state, previous_tree_noise=tree_noise, tree_state=new_tree_state)
    return noised_sample, new_global_state

  @classmethod
  def build_l2_gaussian_query(cls,
                              clip_norm,
                              noise_multiplier,
                              record_specs,
                              noise_seed=None,
                              use_efficient=True):
    """Returns `TreeResidualSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      clip_norm: Each record will be clipped so that it has L2 norm at most
        `clip_norm`.
      noise_multiplier: The effective noise multiplier for the sum of records.
        Noise standard deviation is `clip_norm*noise_multiplier`.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_seed: Integer seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    if clip_norm <= 0:
      raise ValueError(f'`clip_norm` must be positive, got {clip_norm}.')

    if noise_multiplier < 0:
      raise ValueError(
          f'`noise_multiplier` must be non-negative, got {noise_multiplier}.')

    gaussian_noise_generator = tree_aggregation.GaussianNoiseGenerator(
        noise_std=clip_norm * noise_multiplier,
        specs=record_specs,
        seed=noise_seed)

    def l2_clip_fn(record_as_list, clip_norm):
      clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_norm)
      return clipped_record

    return cls(
        clip_fn=l2_clip_fn,
        clip_value=clip_norm,
        record_specs=record_specs,
        noise_generator=gaussian_noise_generator,
        use_efficient=use_efficient)

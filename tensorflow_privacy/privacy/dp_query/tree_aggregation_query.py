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
"""`DPQuery`s for differentially private tree aggregation protocols.

`TreeCumulativeSumQuery` and `TreeResidualSumQuery` are `DPQuery`s for continual
online observation queries relying on `tree_aggregation`. 'Online' means that
the leaf nodes of the tree arrive one by one as the time proceeds. The core
logic of tree aggregation is implemented in `tree_aggregation.TreeAggregator`
and `tree_aggregation.EfficientTreeAggregator`.

Depending on the data streaming setting (single/multi-pass), the privacy
accounting method ((epsilon,delta)-DP/RDP/zCDP), and the restart strategy (see
`restart_query`), the DP bound can be computed by one of the public methods
in `analysis.tree_aggregation_accountant`.

For example, for a single-pass algorithm where a sample may appear at most once
in the querying process; if `get_noised_result` is called `steps` times, the
corresponding epsilon for a `target_delta` and `noise_multiplier` to achieve
(epsilon,delta)-DP can be computed as:
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp_tree_restart(noise_multiplier, [steps], orders)
  eps = rdp_accountant.get_privacy_spent(orders, rdp, target_delta)[0]
"""

import attr
import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation


# TODO(b/193679963): define `RestartQuery` and move `RestartIndicator` to be
# in the same module.


class TreeCumulativeSumQuery(dp_query.SumAggregationDPQuery):
  """Returns private cumulative sums by clipping and adding correlated noise.

  Consider calling `get_noised_result` T times, and each (x_i, i=0,2,...,T-1) is
  the private value returned by `accumulate_record`, i.e. x_i = sum_{j=0}^{n-1}
  x_{i,j} where each x_{i,j} is a private record in the database. This class is
  intended to make multiple queries, which release privatized values of the
  cumulative sums s_i = sum_{k=0}^{i} x_k, for i=0,...,T-1.
  Each call to `get_noised_result` releases the next cumulative sum s_i, which
  is in contrast to the GaussianSumQuery that releases x_i. Noise for the
  cumulative sums is accomplished using the tree aggregation logic in
  `tree_aggregation`, which is proportional to log(T).

  Example usage:
    query = TreeCumulativeSumQuery(...)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i, samples in enumerate(streaming_samples):
      sample_state = query.initial_sample_state(samples[0])
      # Compute  x_i = sum_{j=0}^{n-1} x_{i,j}
      for j,sample in enumerate(samples):
        sample_state = query.accumulate_record(params, sample_state, sample)
      # noised_cumsum is privatized estimate of s_i
      noised_cumsum, global_state, event = query.get_noised_result(
        sample_state, global_state)

  Attributes:
    clip_fn: Callable that specifies clipping function. `clip_fn` receives two
      arguments: a flat list of vars in a record and a `clip_value` to clip the
        corresponding record, e.g. clip_fn(flat_record, clip_value).
    clip_value: float indicating the value at which to clip the record.
    record_specs: `Collection[tf.TensorSpec]` specifying shapes of records.
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
    return TreeCumulativeSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        samples_cumulative_sum=initial_samples_cumulative_sum)

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
    noised_cumulative_sum = tf.nest.map_structure(tf.add, new_cumulative_sum,
                                                  cumulative_sum_noise)
    new_global_state = attr.evolve(
        global_state,
        samples_cumulative_sum=new_cumulative_sum,
        tree_state=new_tree_state)
    event = dp_accounting.UnsupportedDpEvent()
    return noised_cumulative_sum, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met.

    Args:
      noised_results: Noised cumulative sum returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        has current sample's cumulative sum and tree state for the next
        cumulative sum.

    Returns:
      New global state with current noised cumulative sum and restarted tree
        state for the next cumulative sum.
    """
    new_tree_state = self._tree_aggregator.reset_state(global_state.tree_state)
    return attr.evolve(
        global_state,
        samples_cumulative_sum=noised_results,
        tree_state=new_tree_state)

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
        Noise standard deviation is `clip_norm*noise_multiplier`. The value can
        be used as the input of the privacy accounting functions in
        `analysis.tree_aggregation_accountant`.
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
  """Implements DPQuery for adding correlated noise through tree structure.

  Clips and sums records in current sample x_i = sum_{j=0}^{n-1} x_{i,j};
  returns the current sample adding the noise residual from tree aggregation.
  The returned value is conceptually equivalent to the following: calculates
  cumulative sum of samples over time s_i = sum_{k=0}^i x_i (instead of only
  current sample) with added noise by tree aggregation protocol that is
  proportional to log(T), T being the number of times the query is called; r
  eturns the residual between the current noised cumsum noised(s_i) and the
  previous one noised(s_{i-1}) when the query is called.

  This can be used as a drop-in replacement for `GaussianSumQuery`, and can
  offer stronger utility/privacy tradeoffs when aplification-via-sampling is not
  possible, or when privacy epsilon is relativly large.  This may result in
  more noise by a log(T) factor in each individual estimate of x_i, but if the
  x_i are used in the underlying code to compute cumulative sums, the noise in
  those sums can be less. That is, this allows us to adapt code that was written
  to use a regular `SumQuery` to benefit from the tree aggregation protocol.

  Combining this query with a SGD optimizer can be used to implement the
  DP-FTRL algorithm in
  "Practical and Private (Deep) Learning without Sampling or Shuffling".

  Example usage:
    query = TreeResidualSumQuery(...)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i, samples in enumerate(streaming_samples):
      sample_state = query.initial_sample_state(samples[0])
      # Compute  x_i = sum_{j=0}^{n-1} x_{i,j}
      for j,sample in enumerate(samples):
        sample_state = query.accumulate_record(params, sample_state, sample)
      # noised_sum is privatized estimate of x_i by conceptually postprocessing
      # noised cumulative sum s_i
      noised_sum, global_state, event = query.get_noised_result(
        sample_state, global_state)

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

  def _zero_initial_noise(self):
    return tf.nest.map_structure(lambda spec: tf.zeros(spec.shape),
                                 self._record_specs)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    initial_tree_state = self._tree_aggregator.init_state()
    return TreeResidualSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        previous_tree_noise=self._zero_initial_noise())

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.clip_value

  def preprocess_record_l2_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(record, clipped_as_list), norm

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
    event = dp_accounting.UnsupportedDpEvent()
    return noised_sample, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met.

    Args:
      noised_results: Noised results returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        records noise for the conceptual cumulative sum of the current leaf
        node, and tree state for the next conceptual cumulative sum.

    Returns:
      New global state with zero noise and restarted tree state.
    """
    del noised_results
    new_tree_state = self._tree_aggregator.reset_state(global_state.tree_state)
    return attr.evolve(
        global_state,
        previous_tree_noise=self._zero_initial_noise(),
        tree_state=new_tree_state)

  def reset_l2_clip_gaussian_noise(self, global_state, clip_norm, stddev):
    noise_generator_state = global_state.tree_state.value_generator_state
    assert isinstance(self._tree_aggregator.value_generator,
                      tree_aggregation.GaussianNoiseGenerator)
    noise_generator_state = self._tree_aggregator.value_generator.make_state(
        noise_generator_state.seeds, stddev)
    new_tree_state = attr.evolve(
        global_state.tree_state, value_generator_state=noise_generator_state)
    return attr.evolve(
        global_state, clip_value=clip_norm, tree_state=new_tree_state)

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
        Noise standard deviation is `clip_norm*noise_multiplier`. The value can
        be used as the input of the privacy accounting functions in
        `analysis.tree_aggregation_accountant`.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_seed: Integer seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    if clip_norm < 0:
      raise ValueError(f'`clip_norm` must be non-negative, got {clip_norm}.')

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

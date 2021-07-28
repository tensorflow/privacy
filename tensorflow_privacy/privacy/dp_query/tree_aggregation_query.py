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
the leaf nodes of the tree arrive one by one as the time proceeds.

`TreeRangeSumQuery` is a `DPQuery`s for offline tree aggregation protocol.
'Offline' means all the leaf nodes are ready before the protocol starts.
"""
import math

import attr
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import distributed_discrete_gaussian_query
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query
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


@tf.function
def _build_tree_from_leaf(leaf_nodes: tf.Tensor, arity: int) -> tf.RaggedTensor:
  """A function constructs a complete tree given all the leaf nodes.

  The function takes a 1-D array representing the leaf nodes of a tree and the
  tree's arity, and constructs a complete tree by recursively summing the
  adjacent children to get the parent until reaching the root node. Because we
  assume a complete tree, if the number of leaf nodes does not divide arity, the
  leaf nodes will be padded with zeros.

  Args:
    leaf_nodes: A 1-D array storing the leaf nodes of the tree.
    arity: A `int` for the branching factor of the tree, i.e. the number of
      children for each internal node.

  Returns:
    `tf.RaggedTensor` representing the tree. For example, if
    `leaf_nodes=tf.Tensor([1, 2, 3, 4])` and `arity=2`, then the returned value
    should be `tree=tf.RaggedTensor([[10],[3,7],[1,2,3,4]])`. In this way,
    `tree[layer][index]` can be used to access the node indexed by (layer,
    index) in the tree,
  """

  def pad_zero(leaf_nodes, size):
    paddings = [[0, size - len(leaf_nodes)]]
    return tf.pad(leaf_nodes, paddings)

  leaf_nodes_size = tf.constant(len(leaf_nodes), dtype=tf.float32)
  num_layers = tf.math.ceil(
      tf.math.log(leaf_nodes_size) /
      tf.math.log(tf.cast(arity, dtype=tf.float32))) + 1
  leaf_nodes = pad_zero(
      leaf_nodes, tf.math.pow(tf.cast(arity, dtype=tf.float32), num_layers - 1))

  def _shrink_layer(layer: tf.Tensor, arity: int) -> tf.Tensor:
    return tf.reduce_sum((tf.reshape(layer, (-1, arity))), 1)

  # The following `tf.while_loop` constructs the tree from bottom up by
  # iteratively applying `_shrink_layer` to each layer of the tree. The reason
  # for the choice of TF1.0-style `tf.while_loop` is that @tf.function does not
  # support auto-translation from python loop to tf loop when loop variables
  # contain a `RaggedTensor` whose shape changes across iterations.

  idx = tf.identity(num_layers)
  loop_cond = lambda i, h: tf.less_equal(2.0, i)

  def _loop_body(i, h):
    return [
        tf.add(i, -1.0),
        tf.concat(([_shrink_layer(h[0], arity)], h), axis=0)
    ]

  _, tree = tf.while_loop(
      loop_cond,
      _loop_body, [idx, tf.RaggedTensor.from_tensor([leaf_nodes])],
      shape_invariants=[
          idx.get_shape(),
          tf.RaggedTensorSpec(dtype=leaf_nodes.dtype, ragged_rank=1)
      ])

  return tree


class TreeRangeSumQuery(dp_query.SumAggregationDPQuery):
  """Implements dp_query for accurate range queries using tree aggregation.

  Implements a variant of the tree aggregation protocol from. "Is interaction
  necessary for distributed private learning?. Adam Smith, Abhradeep Thakurta,
  Jalaj Upadhyay." Builds a tree on top of the input record and adds noise to
  the tree for differential privacy. Any range query can be decomposed into the
  sum of O(log(n)) nodes in the tree compared to O(n) when using a histogram.
  Improves efficiency and reduces noise scale.
  """

  @attr.s(frozen=True)
  class GlobalState(object):
    """Class defining global state for TreeRangeSumQuery.

    Attributes:
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has).
      inner_query_state: The global state of the inner query.
    """
    arity = attr.ib()
    inner_query_state = attr.ib()

  def __init__(self,
               inner_query: dp_query.SumAggregationDPQuery,
               arity: int = 2):
    """Initializes the `TreeRangeSumQuery`.

    Args:
      inner_query: The inner `DPQuery` that adds noise to the tree.
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has). Defaults to 2.
    """
    self._inner_query = inner_query
    self._arity = arity

    if self._arity < 1:
      raise ValueError(f'Invalid arity={arity} smaller than 2.')

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return TreeRangeSumQuery.GlobalState(
        arity=self._arity,
        inner_query_state=self._inner_query.initial_global_state())

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return (global_state.arity,
            self._inner_query.derive_sample_params(
                global_state.inner_query_state))

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    This method builds the tree, flattens it and applies
    `inner_query.preprocess_record` to the flattened tree.

    Args:
      params: Hyper-parameters for preprocessing record.
      record: A histogram representing the leaf nodes of the tree.

    Returns:
      A `tf.Tensor` representing the flattened version of the preprocessed tree.
    """
    arity, inner_query_params = params
    preprocessed_record = _build_tree_from_leaf(record, arity).flat_values
    preprocessed_record = self._inner_query.preprocess_record(
        inner_query_params, preprocessed_record)

    # The following codes reshape the output vector so the output shape of can
    # be statically inferred. This is useful when used with
    # `tff.aggregators.DifferentiallyPrivateFactory` because it needs to know
    # the output shape of this function statically and explicitly.
    preprocessed_record_shape = [
        (self._arity**(math.ceil(math.log(record.shape[0], self._arity)) + 1) -
         1) // (self._arity - 1)
    ]
    return tf.reshape(preprocessed_record, preprocessed_record_shape)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    This function re-constructs the `tf.RaggedTensor` from the flattened tree
    output by `preprocess_records.`

    Args:
      sample_state: A `tf.Tensor` for the flattened tree.
      global_state: The global state of the protocol.

    Returns:
      A `tf.RaggedTensor` representing the tree.
    """
    # The [0] is needed because of how tf.RaggedTensor.from_two_splits works.
    # print(tf.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6],
    #                                       row_splits=[0, 4, 4, 7, 8, 8]))
    # <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
    # This part is not written in tensorflow and will be executed on the server
    # side instead of the client side if used with
    # tff.aggregators.DifferentiallyPrivateFactory for federated learning.
    row_splits = [0] + [
        (self._arity**(x + 1) - 1) // (self._arity - 1) for x in range(
            math.floor(math.log(sample_state.shape[0], self._arity)) + 1)
    ]
    tree = tf.RaggedTensor.from_row_splits(
        values=sample_state, row_splits=row_splits)
    return tree, global_state

  @classmethod
  def build_central_gaussian_query(cls,
                                   l2_norm_clip: float,
                                   stddev: float,
                                   arity: int = 2):
    """Returns `TreeRangeSumQuery` with central Gaussian noise.

    Args:
      l2_norm_clip: Each record should be clipped so that it has L2 norm at most
        `l2_norm_clip`.
      stddev: Stddev of the central Gaussian noise.
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has). Defaults to 2.
    """
    if l2_norm_clip <= 0:
      raise ValueError(f'`l2_norm_clip` must be positive, got {l2_norm_clip}.')

    if stddev < 0:
      raise ValueError(f'`stddev` must be non-negative, got {stddev}.')

    if arity < 2:
      raise ValueError(f'`arity` must be at least 2, got {arity}.')

    inner_query = gaussian_query.GaussianSumQuery(l2_norm_clip, stddev)

    return cls(arity=arity, inner_query=inner_query)

  @classmethod
  def build_distributed_discrete_gaussian_query(cls,
                                                l2_norm_bound: float,
                                                local_stddev: float,
                                                arity: int = 2):
    """Returns `TreeRangeSumQuery` with central Gaussian noise.

    Args:
      l2_norm_bound: Each record should be clipped so that it has L2 norm at
        most `l2_norm_bound`.
      local_stddev: Scale/stddev of the local discrete Gaussian noise.
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has). Defaults to 2.
    """
    if l2_norm_bound <= 0:
      raise ValueError(
          f'`l2_clip_bound` must be positive, got {l2_norm_bound}.')

    if local_stddev < 0:
      raise ValueError(
          f'`local_stddev` must be non-negative, got {local_stddev}.')

    if arity < 2:
      raise ValueError(f'`arity` must be at least 2, got {arity}.')

    inner_query = distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery(
        l2_norm_bound, local_stddev)

    return cls(arity=arity, inner_query=inner_query)

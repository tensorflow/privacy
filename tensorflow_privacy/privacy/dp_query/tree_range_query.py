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
"""`DPQuery`s for offline differentially private tree aggregation protocols.

'Offline' means all the leaf nodes are ready before the protocol starts.
"""

import distutils
import math
from typing import Optional

import attr
import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import distributed_discrete_gaussian_query
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query


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
    paddings = tf.zeros(
        shape=(size - leaf_nodes.shape[0],), dtype=leaf_nodes.dtype)
    return tf.concat((leaf_nodes, paddings), axis=0)

  leaf_nodes_size = tf.constant(leaf_nodes.shape[0], dtype=tf.float32)
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

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self.preprocess_record(
        self.derive_sample_params(self.initial_global_state()),
        super().initial_sample_state(template))

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
    # The following codes reshape the output vector so the output shape of can
    # be statically inferred. This is useful when used with
    # `tff.aggregators.DifferentiallyPrivateFactory` because it needs to know
    # the output shape of this function statically and explicitly.
    preprocessed_record_shape = [
        (self._arity**(math.ceil(math.log(record.shape[0], self._arity)) + 1) -
         1) // (self._arity - 1)
    ]
    preprocessed_record = tf.reshape(preprocessed_record,
                                     preprocessed_record_shape)
    preprocessed_record = self._inner_query.preprocess_record(
        inner_query_params, preprocessed_record)

    return preprocessed_record

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
    sample_state, inner_query_state, _ = self._inner_query.get_noised_result(
        sample_state, global_state.inner_query_state)
    new_global_state = TreeRangeSumQuery.GlobalState(
        arity=global_state.arity, inner_query_state=inner_query_state)

    row_splits = [0] + [
        (self._arity**(x + 1) - 1) // (self._arity - 1) for x in range(
            math.floor(math.log(sample_state.shape[0], self._arity)) + 1)
    ]
    tree = tf.RaggedTensor.from_row_splits(
        values=sample_state, row_splits=row_splits)
    event = dp_accounting.UnsupportedDpEvent()
    return tree, new_global_state, event

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


def _get_add_noise(stddev, seed: Optional[int] = None):
  """Utility function to decide which `add_noise` to use according to tf version."""
  if distutils.version.LooseVersion(
      tf.__version__) < distutils.version.LooseVersion('2.0.0'):

    # The seed should be only used for testing purpose.
    if seed is not None:
      tf.random.set_seed(seed)

    def add_noise(v):
      return v + tf.random.normal(
          tf.shape(input=v), stddev=stddev, dtype=v.dtype)
  else:
    random_normal = tf.random_normal_initializer(stddev=stddev, seed=seed)

    def add_noise(v):
      return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)

  return add_noise

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
"""Tests for `tree_range_query`."""

import math

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import tree_range_query


class BuildTreeTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      leaf_nodes_size=[1, 2, 3, 4, 5],
      arity=[2, 3],
      dtype=[tf.int32, tf.float32],
  )
  def test_build_tree_from_leaf(self, leaf_nodes_size, arity, dtype):
    """Test whether `_build_tree_from_leaf` will output the correct tree."""

    leaf_nodes = tf.cast(tf.range(leaf_nodes_size), dtype)
    depth = math.ceil(math.log(leaf_nodes_size, arity)) + 1

    tree = tree_range_query._build_tree_from_leaf(leaf_nodes, arity)

    self.assertEqual(depth, tree.shape[0])

    for layer in range(depth):
      reverse_depth = tree.shape[0] - layer - 1
      span_size = arity**reverse_depth
      for idx in range(arity**layer):
        left = idx * span_size
        right = (idx + 1) * span_size
        expected_value = sum(leaf_nodes[left:right])
        self.assertEqual(tree[layer][idx], expected_value)


class TreeRangeSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      inner_query=['central', 'distributed'],
      params=[(0., 1., 2), (1., -1., 2), (1., 1., 1)],
  )
  def test_raises_error(self, inner_query, params):
    clip_norm, stddev, arity = params
    with self.assertRaises(ValueError):
      if inner_query == 'central':
        tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
            clip_norm, stddev, arity)
      elif inner_query == 'distributed':
        tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
            clip_norm, stddev, arity)

  @parameterized.product(
      inner_query=['central', 'distributed'],
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0])
  def test_initial_global_state_type(self, inner_query, clip_norm, stddev):

    if inner_query == 'central':
      query = tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
          clip_norm, stddev)
    elif inner_query == 'distributed':
      query = tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          clip_norm, stddev)
    global_state = query.initial_global_state()
    self.assertIsInstance(global_state,
                          tree_range_query.TreeRangeSumQuery.GlobalState)

  @parameterized.product(
      inner_query=['central', 'distributed'],
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0],
      arity=[2, 3, 4])
  def test_derive_sample_params(self, inner_query, clip_norm, stddev, arity):
    if inner_query == 'central':
      query = tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
          clip_norm, stddev, arity)
    elif inner_query == 'distributed':
      query = tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          clip_norm, stddev, arity)
    global_state = query.initial_global_state()
    derived_arity, inner_query_state = query.derive_sample_params(global_state)
    self.assertAllClose(derived_arity, arity)
    if inner_query == 'central':
      self.assertAllClose(inner_query_state, clip_norm)
    elif inner_query == 'distributed':
      self.assertAllClose(inner_query_state.l2_norm_bound, clip_norm)
      self.assertAllClose(inner_query_state.local_stddev, stddev)

  @parameterized.product(
      (dict(arity=2, expected_tree=[1, 1, 0, 1, 0, 0, 0]),
       dict(arity=3, expected_tree=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])),
      inner_query=['central', 'distributed'],
  )
  def test_preprocess_record(self, inner_query, arity, expected_tree):
    if inner_query == 'central':
      query = tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
      expected_tree = tf.cast(expected_tree, tf.float32)
    elif inner_query == 'distributed':
      query = tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.int32)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    self.assertAllClose(preprocessed_record, expected_tree)

  @parameterized.named_parameters(
      ('stddev_1', 1, tf.constant([1, 0], dtype=tf.int32), [1, 1, 0]),
      ('stddev_0_1', 4, tf.constant([1, 0], dtype=tf.int32), [1, 1, 0]),
  )
  def test_distributed_preprocess_record_with_noise(self, local_stddev, record,
                                                    expected_tree):
    query = tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
        10., local_stddev)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)

    preprocessed_record = query.preprocess_record(params, record)

    self.assertAllClose(
        preprocessed_record, expected_tree, atol=10 * local_stddev)

  @parameterized.product(
      (dict(
          arity=2,
          expected_tree=tf.ragged.constant([[1], [1, 0], [1, 0, 0, 0]])),
       dict(
           arity=3,
           expected_tree=tf.ragged.constant([[1], [1, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0]]))),
      inner_query=['central', 'distributed'],
  )
  def test_get_noised_result(self, inner_query, arity, expected_tree):
    if inner_query == 'central':
      query = tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
      expected_tree = tf.cast(expected_tree, tf.float32)
    elif inner_query == 'distributed':
      query = tree_range_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.int32)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state, _ = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)

  @parameterized.product(stddev=[0.1, 1.0, 10.0])
  def test_central_get_noised_result_with_noise(self, stddev):
    query = tree_range_query.TreeRangeSumQuery.build_central_gaussian_query(
        10., stddev)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, tf.constant([1., 0.]))
    sample_state, global_state, _ = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(
        sample_state, tf.ragged.constant([[1.], [1., 0.]]), atol=10 * stddev)


if __name__ == '__main__':
  tf.test.main()

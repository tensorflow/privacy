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
"""Tests for `tree_aggregation_query`."""

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import test_utils
from tensorflow_privacy.privacy.dp_query import tree_aggregation
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query

STRUCT_RECORD = [
    tf.constant([[2.0, 0.0], [0.0, 1.0]]),
    tf.constant([-1.0, 0.0])
]

SINGLE_VALUE_RECORDS = [tf.constant(1.), tf.constant(3.), tf.constant(5.)]

STRUCTURE_SPECS = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                        STRUCT_RECORD)
NOISE_STD = 5.0

STREAMING_SCALARS = np.array(range(7), dtype=np.single)


def _get_noise_generator(specs, stddev=NOISE_STD, seed=1):
  return tree_aggregation.GaussianNoiseGenerator(
      noise_std=stddev, specs=specs, seed=seed)


def _get_noise_fn(specs, stddev=NOISE_STD, seed=1):
  random_generator = tf.random.Generator.from_seed(seed)

  def noise_fn():
    shape = tf.nest.map_structure(lambda spec: spec.shape, specs)
    return tf.nest.map_structure(
        lambda x: random_generator.normal(x, stddev=stddev), shape)

  return noise_fn


def _get_no_noise_fn(specs):
  shape = tf.nest.map_structure(lambda spec: spec.shape, specs)

  def no_noise_fn():
    return tf.nest.map_structure(tf.zeros, shape)

  return no_noise_fn


def _get_l2_clip_fn():

  def l2_clip_fn(record_as_list, clip_value):
    clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_value)
    return clipped_record

  return l2_clip_fn


def _get_l_infty_clip_fn():

  def l_infty_clip_fn(record_as_list, clip_value):

    def clip(record):
      return tf.clip_by_value(
          record, clip_value_min=-clip_value, clip_value_max=clip_value)

    clipped_record = tf.nest.map_structure(clip, record_as_list)
    return clipped_record

  return l_infty_clip_fn


class TreeCumulativeSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_correct_initial_global_state_struct_type(self):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=10.,
        noise_generator=_get_no_noise_fn(STRUCTURE_SPECS),
        record_specs=STRUCTURE_SPECS)
    global_state = query.initial_global_state()

    self.assertIsInstance(global_state.tree_state, tree_aggregation.TreeState)
    expected_cum_sum = tf.nest.map_structure(lambda spec: tf.zeros(spec.shape),
                                             STRUCTURE_SPECS)
    self.assertAllClose(expected_cum_sum, global_state.samples_cumulative_sum)

  def test_correct_initial_global_state_single_value_type(self):
    record_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                         SINGLE_VALUE_RECORDS[0])
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=10.,
        noise_generator=_get_no_noise_fn(record_specs),
        record_specs=record_specs)
    global_state = query.initial_global_state()

    self.assertIsInstance(global_state.tree_state, tree_aggregation.TreeState)
    expected_cum_sum = tf.nest.map_structure(lambda spec: tf.zeros(spec.shape),
                                             record_specs)
    self.assertAllClose(expected_cum_sum, global_state.samples_cumulative_sum)

  @parameterized.named_parameters(
      ('not_clip_single_record', SINGLE_VALUE_RECORDS[0], 10.0),
      ('clip_single_record', SINGLE_VALUE_RECORDS[1], 1.0))
  def test_l2_clips_single_record(self, record, l2_norm_clip):
    record_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                         SINGLE_VALUE_RECORDS[0])
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=l2_norm_clip,
        noise_generator=_get_no_noise_fn(record_specs),
        record_specs=record_specs)
    global_state = query.initial_global_state()
    record_norm = tf.norm(record)
    if record_norm > l2_norm_clip:
      expected_clipped_record = tf.nest.map_structure(
          lambda t: t * l2_norm_clip / record_norm, record)
    else:
      expected_clipped_record = record
    clipped_record = query.preprocess_record(global_state.clip_value, record)
    self.assertAllClose(expected_clipped_record, clipped_record)

  @parameterized.named_parameters(
      ('not_clip_structure_record', STRUCT_RECORD, 10.0),
      ('clip_structure_record', STRUCT_RECORD, 1.0))
  def test_l2_clips_structure_type_record(self, record, l2_norm_clip):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=l2_norm_clip,
        noise_generator=_get_no_noise_fn(STRUCTURE_SPECS),
        record_specs=tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                           record))
    global_state = query.initial_global_state()
    record_norm = tf.linalg.global_norm(record)
    if record_norm > l2_norm_clip:
      expected_clipped_record = tf.nest.map_structure(
          lambda t: t * l2_norm_clip / record_norm, record)
    else:
      expected_clipped_record = record
    clipped_record = query.preprocess_record(global_state.clip_value, record)
    self.assertAllClose(expected_clipped_record, clipped_record)

  @parameterized.named_parameters(
      ('not_clip_single_record', SINGLE_VALUE_RECORDS[0], 10.0),
      ('clip_single_record', SINGLE_VALUE_RECORDS[1], 1.0))
  def test_l_infty_clips_single_record(self, record, norm_clip):
    record_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                         SINGLE_VALUE_RECORDS[0])
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l_infty_clip_fn(),
        clip_value=norm_clip,
        noise_generator=_get_no_noise_fn(record_specs),
        record_specs=record_specs)
    global_state = query.initial_global_state()
    expected_clipped_record = tf.nest.map_structure(
        lambda t: tf.clip_by_value(t, -norm_clip, norm_clip), record)
    clipped_record = query.preprocess_record(global_state.clip_value, record)
    self.assertAllClose(expected_clipped_record, clipped_record)

  @parameterized.named_parameters(
      ('not_clip_structure_record', STRUCT_RECORD, 10.0),
      ('clip_structure_record', STRUCT_RECORD, 1.0))
  def test_linfty_clips_structure_type_record(self, record, norm_clip):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l_infty_clip_fn(),
        clip_value=norm_clip,
        noise_generator=_get_no_noise_fn(STRUCTURE_SPECS),
        record_specs=tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                           record))
    global_state = query.initial_global_state()
    expected_clipped_record = tf.nest.map_structure(
        lambda t: tf.clip_by_value(t, -norm_clip, norm_clip), record)
    clipped_record = query.preprocess_record(global_state.clip_value, record)
    self.assertAllClose(expected_clipped_record, clipped_record)

  def test_noiseless_query_single_value_type_record(self):
    record_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                         SINGLE_VALUE_RECORDS[0])
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=10.,
        noise_generator=_get_no_noise_fn(record_specs),
        record_specs=record_specs)
    query_result, _ = test_utils.run_query(query, SINGLE_VALUE_RECORDS)
    expected = tf.constant(9.)
    self.assertAllClose(query_result, expected)

  def test_noiseless_query_structure_type_record(self):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=10.,
        noise_generator=_get_no_noise_fn(STRUCTURE_SPECS),
        record_specs=STRUCTURE_SPECS)
    query_result, _ = test_utils.run_query(query,
                                           [STRUCT_RECORD, STRUCT_RECORD])
    expected = tf.nest.map_structure(lambda a, b: a + b, STRUCT_RECORD,
                                     STRUCT_RECORD)
    self.assertAllClose(query_result, expected)

  @parameterized.named_parameters(
      ('two_records_noise_fn', [2.71828, 3.14159], _get_noise_fn),
      ('five_records_noise_fn', np.random.uniform(size=5).tolist(),
       _get_noise_fn),
      ('two_records_generator', [2.71828, 3.14159], _get_noise_generator),
      ('five_records_generator', np.random.uniform(size=5).tolist(),
       _get_noise_generator),
  )
  def test_noisy_cumsum_and_state_update(self, records, value_generator):
    num_trials = 200
    record_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(tf.shape(t)),
                                         records[0])
    noised_sums = []
    for i in range(num_trials):
      query = tree_aggregation_query.TreeCumulativeSumQuery(
          clip_fn=_get_l2_clip_fn(),
          clip_value=10.,
          noise_generator=value_generator(record_specs, seed=i),
          record_specs=record_specs)
      query_result, _ = test_utils.run_query(query, records)
      noised_sums.append(query_result)
    result_stddev = np.std(noised_sums)
    self.assertNear(result_stddev, NOISE_STD, 0.7)  # value for chi-squared test

  @parameterized.named_parameters(
      ('no_clip', STREAMING_SCALARS, 10., np.cumsum(STREAMING_SCALARS)),
      ('all_clip', STREAMING_SCALARS, 0.5, STREAMING_SCALARS * 0.5),
      # STREAMING_SCALARS is list(range(7)), only the last element is clipped
      # for the following test, which makes the expected value for the last sum
      # to be `cumsum(5)+5`.
      ('partial_clip', STREAMING_SCALARS, 5.,
       np.append(np.cumsum(STREAMING_SCALARS[:-1]), 20.)),
  )
  def test_partial_sum_scalar_no_noise(self, streaming_scalars, clip_norm,
                                       partial_sum):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=clip_norm,
        noise_generator=lambda: 0.,
        record_specs=tf.TensorSpec([]),
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for scalar, expected_sum in zip(streaming_scalars, partial_sum):
      sample_state = query.initial_sample_state(scalar)
      sample_state = query.accumulate_record(params, sample_state, scalar)
      query_result, global_state = query.get_noised_result(
          sample_state, global_state)
      self.assertEqual(query_result, expected_sum)

  @parameterized.named_parameters(
      ('s0t1step8', 0., 1., [1., 1., 2., 1., 2., 2., 3., 1.]),
      ('s1t1step8', 1., 1., [2., 3., 5., 5., 7., 8., 10., 9.]),
      ('s1t2step8', 1., 2., [3., 4., 7., 6., 9., 10., 13., 10.]),
  )
  def test_partial_sum_scalar_tree_aggregation(self, scalar_value,
                                               tree_node_value,
                                               expected_values):
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False,
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for val in expected_values:
      # For each streaming step i , the expected value is roughly
      # `scalar_value*i + tree_aggregation(tree_node_value, i)`
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state = query.get_noised_result(
          sample_state, global_state)
      self.assertEqual(query_result, val)

  @parameterized.named_parameters(
      ('efficient', True, tree_aggregation.EfficientTreeAggregator),
      ('normal', False, tree_aggregation.TreeAggregator),
  )
  def test_sum_tree_aggregator_instance(self, use_efficient, tree_class):
    specs = tf.TensorSpec([])
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=1.,
        noise_generator=_get_noise_fn(specs, 1.),
        record_specs=specs,
        use_efficient=use_efficient,
    )
    self.assertIsInstance(query._tree_aggregator, tree_class)

  @parameterized.named_parameters(
      ('r5d10n0s1s16eff', 5, 10, 0., 1, 16, 0.1, True),
      ('r3d5n1s1s32eff', 3, 5, 1., 1, 32, 1., True),
      ('r10d3n1s2s16eff', 10, 3, 1., 2, 16, 10., True),
      ('r10d3n1s2s16', 10, 3, 1., 2, 16, 10., False),
  )
  def test_build_l2_gaussian_query(self, records_num, record_dim,
                                   noise_multiplier, seed, total_steps, clip,
                                   use_efficient):
    record_specs = tf.TensorSpec(shape=[record_dim])
    query = tree_aggregation_query.TreeCumulativeSumQuery.build_l2_gaussian_query(
        clip_norm=clip,
        noise_multiplier=noise_multiplier,
        record_specs=record_specs,
        noise_seed=seed,
        use_efficient=use_efficient)
    reference_query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=clip,
        noise_generator=_get_noise_generator(record_specs,
                                             clip * noise_multiplier, seed),
        record_specs=record_specs,
        use_efficient=use_efficient)
    global_state = query.initial_global_state()
    reference_global_state = reference_query.initial_global_state()

    for _ in range(total_steps):
      records = [
          tf.random.uniform(shape=[record_dim], maxval=records_num)
          for _ in range(records_num)
      ]
      query_result, global_state = test_utils.run_query(query, records,
                                                        global_state)
      reference_query_result, reference_global_state = test_utils.run_query(
          reference_query, records, reference_global_state)
      self.assertAllClose(query_result, reference_query_result, rtol=1e-6)


class TreeResidualQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('r5d10n0s1s16eff', 5, 10, 0., 1, 16, 0.1, True),
      ('r3d5n1s1s32eff', 3, 5, 1., 1, 32, 1., True),
      ('r10d3n1s2s16eff', 10, 3, 1., 2, 16, 10., True),
      ('r10d3n1s2s16', 10, 3, 1., 2, 16, 10., False),
  )
  def test_sum(self, records_num, record_dim, noise_multiplier, seed,
               total_steps, clip, use_efficient):
    record_specs = tf.TensorSpec(shape=[record_dim])
    query = tree_aggregation_query.TreeResidualSumQuery.build_l2_gaussian_query(
        clip_norm=clip,
        noise_multiplier=noise_multiplier,
        record_specs=record_specs,
        noise_seed=seed,
        use_efficient=use_efficient)
    sum_query = tree_aggregation_query.TreeCumulativeSumQuery.build_l2_gaussian_query(
        clip_norm=clip,
        noise_multiplier=noise_multiplier,
        record_specs=record_specs,
        noise_seed=seed,
        use_efficient=use_efficient)
    global_state = query.initial_global_state()
    sum_global_state = sum_query.initial_global_state()

    cumsum_result = tf.zeros(shape=[record_dim])
    for _ in range(total_steps):
      records = [
          tf.random.uniform(shape=[record_dim], maxval=records_num)
          for _ in range(records_num)
      ]
      query_result, global_state = test_utils.run_query(query, records,
                                                        global_state)
      sum_query_result, sum_global_state = test_utils.run_query(
          sum_query, records, sum_global_state)
      cumsum_result += query_result
      self.assertAllClose(cumsum_result, sum_query_result, rtol=1e-6)

  @parameterized.named_parameters(
      ('efficient', True, tree_aggregation.EfficientTreeAggregator),
      ('normal', False, tree_aggregation.TreeAggregator),
  )
  def test_sum_tree_aggregator_instance(self, use_efficient, tree_class):
    specs = tf.TensorSpec([])
    query = tree_aggregation_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=1.,
        noise_generator=_get_noise_fn(specs, 1.),
        record_specs=specs,
        use_efficient=use_efficient,
    )
    self.assertIsInstance(query._tree_aggregator, tree_class)


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

    tree = tree_aggregation_query._build_tree_from_leaf(leaf_nodes, arity)

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
        tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
            clip_norm, stddev, arity)
      elif inner_query == 'distributed':
        tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
            clip_norm, stddev, arity)

  @parameterized.product(
      inner_query=['central', 'distributed'],
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0])
  def test_initial_global_state_type(self, inner_query, clip_norm, stddev):

    if inner_query == 'central':
      query = tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
          clip_norm, stddev)
    elif inner_query == 'distributed':
      query = tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          clip_norm, stddev)
    global_state = query.initial_global_state()
    self.assertIsInstance(global_state,
                          tree_aggregation_query.TreeRangeSumQuery.GlobalState)

  @parameterized.product(
      inner_query=['central', 'distributed'],
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0],
      arity=[2, 3, 4])
  def test_derive_sample_params(self, inner_query, clip_norm, stddev, arity):
    if inner_query == 'central':
      query = tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
          clip_norm, stddev, arity)
    elif inner_query == 'distributed':
      query = tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
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
      query = tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
      expected_tree = tf.cast(expected_tree, tf.float32)
    elif inner_query == 'distributed':
      query = tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
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
    query = tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
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
      query = tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
      expected_tree = tf.cast(expected_tree, tf.float32)
    elif inner_query == 'distributed':
      query = tree_aggregation_query.TreeRangeSumQuery.build_distributed_discrete_gaussian_query(
          10., 0., arity)
      record = tf.constant([1, 0, 0, 0], dtype=tf.int32)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)

  @parameterized.product(stddev=[0.1, 1.0, 10.0])
  def test_central_get_noised_result_with_noise(self, stddev):
    query = tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
        10., stddev)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, tf.constant([1., 0.]))
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(
        sample_state, tf.ragged.constant([[1.], [1., 0.]]), atol=10 * stddev)


class CentralTreeSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_initial_global_state_type(self):

    query = tree_aggregation_query.CentralTreeSumQuery(stddev=NOISE_STD)
    global_state = query.initial_global_state()
    self.assertIsInstance(
        global_state, tree_aggregation_query.CentralTreeSumQuery.GlobalState)

  def test_derive_sample_params(self):
    query = tree_aggregation_query.CentralTreeSumQuery(stddev=NOISE_STD)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    self.assertAllClose(params, 10.)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([1, 0, 0, 0], dtype=tf.int32)),
      ('binary_test_float', 2, tf.constant([1., 0., 0., 0.], dtype=tf.float32)),
      ('ternary_test_int', 3, tf.constant([1, 0, 0, 0], dtype=tf.int32)),
      ('ternary_test_float', 3, tf.constant([1., 0., 0., 0.],
                                            dtype=tf.float32)),
  )
  def test_preprocess_record(self, arity, record):
    query = tree_aggregation_query.CentralTreeSumQuery(
        stddev=NOISE_STD, arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)

    self.assertAllClose(preprocessed_record, record)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.constant([5, 5, 0, 0], dtype=tf.int32)),
      ('binary_test_float', 2, tf.constant(
          [10., 10., 0., 0.],
          dtype=tf.float32), tf.constant([5., 5., 0., 0.], dtype=tf.float32)),
      ('ternary_test_int', 3, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.constant([5, 5, 0, 0], dtype=tf.int32)),
      ('ternary_test_float', 3, tf.constant([10., 10., 0., 0.],
                                            dtype=tf.float32),
       tf.constant([5., 5., 0., 0.], dtype=tf.float32)),
  )
  def test_preprocess_record_clipped(self, arity, record,
                                     expected_clipped_value):
    query = tree_aggregation_query.CentralTreeSumQuery(
        stddev=NOISE_STD, arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    self.assertAllClose(preprocessed_record, expected_clipped_value)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[1.], [1., 0.], [1., 0., 0., 0.]])),
      ('binary_test_float', 2, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([[1.], [1., 0.], [1., 0., 0., 0.]])),
      ('ternary_test_int', 3, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[1.], [1., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0., 0., 0.]])),
      ('ternary_test_float', 3, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([[1.], [1., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0., 0., 0.]])),
  )
  def test_get_noised_result(self, arity, record, expected_tree):
    query = tree_aggregation_query.CentralTreeSumQuery(stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)

  @parameterized.named_parameters(
      ('stddev_0_01', 0.01, tf.constant([1, 0], dtype=tf.int32), [1., 1., 0.]),
      ('stddev_0_1', 0.1, tf.constant([1, 0], dtype=tf.int32), [1., 1., 0.]),
  )
  def test_get_noised_result_with_noise(self, stddev, record, expected_tree):
    query = tree_aggregation_query.CentralTreeSumQuery(stddev=stddev, seed=0)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)

    sample_state, _ = query.get_noised_result(preprocessed_record, global_state)

    self.assertAllClose(
        sample_state.flat_values, expected_tree, atol=3 * stddev)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[10.], [10., 0.], [5., 5., 0., 0.]])),
      ('binary_test_float', 2, tf.constant([10., 10., 0., 0.],
                                           dtype=tf.float32),
       tf.ragged.constant([[10.], [10., 0.], [5., 5., 0., 0.]])),
      ('ternary_test_int', 3, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[10.], [10., 0., 0.],
                           [5., 5., 0., 0., 0., 0., 0., 0., 0.]])),
      ('ternary_test_float', 3, tf.constant([10., 10., 0., 0.],
                                            dtype=tf.float32),
       tf.ragged.constant([[10.], [10., 0., 0.],
                           [5., 5., 0., 0., 0., 0., 0., 0., 0.]])),
  )
  def test_get_noised_result_clipped(self, arity, record, expected_tree):
    query = tree_aggregation_query.CentralTreeSumQuery(stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)


class DistributedTreeSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_initial_global_state_type(self):

    query = tree_aggregation_query.DistributedTreeSumQuery(stddev=NOISE_STD)
    global_state = query.initial_global_state()
    self.assertIsInstance(
        global_state,
        tree_aggregation_query.DistributedTreeSumQuery.GlobalState)

  def test_derive_sample_params(self):
    query = tree_aggregation_query.DistributedTreeSumQuery(stddev=NOISE_STD)
    global_state = query.initial_global_state()
    stddev, arity, l1_bound = query.derive_sample_params(global_state)
    self.assertAllClose(stddev, NOISE_STD)
    self.assertAllClose(arity, 2)
    self.assertAllClose(l1_bound, 10)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([1., 1., 0., 1., 0., 0., 0.])),
      ('binary_test_float', 2, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([1., 1., 0., 1., 0., 0., 0.])),
      ('ternary_test_int', 3, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.
                          ])),
      ('ternary_test_float', 3, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.
                          ])),
  )
  def test_preprocess_record(self, arity, record, expected_tree):
    query = tree_aggregation_query.DistributedTreeSumQuery(
        stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    self.assertAllClose(preprocessed_record, expected_tree)

  @parameterized.named_parameters(
      ('stddev_0_01', 0.01, tf.constant([1, 0], dtype=tf.int32), [1., 1., 0.]),
      ('stddev_0_1', 0.1, tf.constant([1, 0], dtype=tf.int32), [1., 1., 0.]),
  )
  def test_preprocess_record_with_noise(self, stddev, record, expected_tree):
    query = tree_aggregation_query.DistributedTreeSumQuery(
        stddev=stddev, seed=0)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)

    preprocessed_record = query.preprocess_record(params, record)

    self.assertAllClose(preprocessed_record, expected_tree, atol=3 * stddev)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant([10., 10., 0., 5., 5., 0., 0.])),
      ('binary_test_float', 2, tf.constant([10., 10., 0., 0.],
                                           dtype=tf.float32),
       tf.ragged.constant([10., 10., 0., 5., 5., 0., 0.])),
      ('ternary_test_int', 3, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant(
           [10., 10., 0., 0., 5., 5., 0., 0., 0., 0., 0., 0., 0.])),
      ('ternary_test_float', 3, tf.constant([10., 10., 0., 0.],
                                            dtype=tf.float32),
       tf.ragged.constant(
           [10., 10., 0., 0., 5., 5., 0., 0., 0., 0., 0., 0., 0.])),
  )
  def test_preprocess_record_clipped(self, arity, record, expected_tree):
    query = tree_aggregation_query.DistributedTreeSumQuery(
        stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    self.assertAllClose(preprocessed_record, expected_tree)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[1.], [1., 0.], [1., 0., 0., 0.]])),
      ('binary_test_float', 2, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([[1.], [1., 0.], [1., 0., 0., 0.]])),
      ('ternary_test_int', 3, tf.constant([1, 0, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[1.], [1., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0., 0., 0.]])),
      ('ternary_test_float', 3, tf.constant([1., 0., 0., 0.], dtype=tf.float32),
       tf.ragged.constant([[1.], [1., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0., 0., 0.]])),
  )
  def test_get_noised_result(self, arity, record, expected_tree):
    query = tree_aggregation_query.DistributedTreeSumQuery(
        stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)

  @parameterized.named_parameters(
      ('binary_test_int', 2, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[10.], [10., 0.], [5., 5., 0., 0.]])),
      ('binary_test_float', 2, tf.constant([10., 10., 0., 0.],
                                           dtype=tf.float32),
       tf.ragged.constant([[10.], [10., 0.], [5., 5., 0., 0.]])),
      ('ternary_test_int', 3, tf.constant([10, 10, 0, 0], dtype=tf.int32),
       tf.ragged.constant([[10.], [10., 0., 0.],
                           [5., 5., 0., 0., 0., 0., 0., 0., 0.]])),
      ('ternary_test_float', 3, tf.constant([10., 10., 0., 0.],
                                            dtype=tf.float32),
       tf.ragged.constant([[10.], [10., 0., 0.],
                           [5., 5., 0., 0., 0., 0., 0., 0., 0.]])),
  )
  def test_get_noised_result_clipped(self, arity, record, expected_tree):
    query = tree_aggregation_query.DistributedTreeSumQuery(
        stddev=0., arity=arity)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state = query.get_noised_result(
        preprocessed_record, global_state)

    self.assertAllClose(sample_state, expected_tree)


if __name__ == '__main__':
  tf.test.main()

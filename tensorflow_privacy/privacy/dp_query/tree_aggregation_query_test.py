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
      ('five_records_noise_fn', np.random.uniform(
          low=0.1, size=5).tolist(), _get_noise_fn),
      ('two_records_generator', [2.71828, 3.14159], _get_noise_generator),
      ('five_records_generator', np.random.uniform(
          low=0.1, size=5).tolist(), _get_noise_generator),
  )
  def test_noisy_cumsum_and_state_update(self, records, value_generator):
    num_trials, vector_size = 10, 100
    record_specs = tf.TensorSpec([vector_size])
    records = [tf.constant(r, shape=[vector_size]) for r in records]
    noised_sums = []
    for i in range(num_trials):
      query = tree_aggregation_query.TreeCumulativeSumQuery(
          clip_fn=_get_l2_clip_fn(),
          clip_value=10.,
          noise_generator=value_generator(record_specs, seed=i),
          record_specs=record_specs)
      query_result, _ = test_utils.run_query(query, records)
      noised_sums.append(query_result.numpy())
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
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
      self.assertEqual(query_result, expected_sum)

  @parameterized.named_parameters(
      ('s0t1', 0., 1.),
      ('s1t1', 1., 1.),
      ('s1t2', 1., 2.),
  )
  def test_partial_sum_scalar_tree_aggregation(self, scalar_value,
                                               tree_node_value):
    total_steps = 8
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False,
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
      # For each streaming step i , the expected value is roughly
      # `scalar_value*(i+1) + tree_aggregation(tree_node_value, i)`.
      # The tree aggregation value can be inferred from the binary
      # representation of the current step.
      self.assertEqual(
          query_result,
          scalar_value * (i + 1) + tree_node_value * bin(i + 1)[2:].count('1'))

  @parameterized.named_parameters(
      ('s0t1f1', 0., 1., 1),
      ('s0t1f2', 0., 1., 2),
      ('s0t1f5', 0., 1., 5),
      ('s1t1f5', 1., 1., 5),
      ('s1t2f2', 1., 2., 2),
      ('s1t5f6', 1., 5., 6),
  )
  def test_sum_scalar_tree_aggregation_reset(self, scalar_value,
                                             tree_node_value, frequency):
    total_steps = 20
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
      if i % frequency == frequency - 1:
        global_state = query.reset_state(query_result, global_state)
      # Expected value is the combination of cumsum of signal; sum of trees
      # that have been reset; current tree sum. The tree aggregation value can
      # be inferred from the binary representation of the current step.
      expected = (
          scalar_value * (i + 1) +
          i // frequency * tree_node_value * bin(frequency)[2:].count('1') +
          tree_node_value * bin(i % frequency + 1)[2:].count('1'))
      self.assertEqual(query_result, expected)

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

  @parameterized.named_parameters(
      ('s0t1f1', 0., 1., 1),
      ('s0t1f2', 0., 1., 2),
      ('s0t1f5', 0., 1., 5),
      ('s1t1f5', 1., 1., 5),
      ('s1t2f2', 1., 2., 2),
      ('s1t5f6', 1., 5., 6),
  )
  def test_scalar_tree_aggregation_reset(self, scalar_value, tree_node_value,
                                         frequency):
    total_steps = 20
    query = tree_aggregation_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
      if i % frequency == frequency - 1:
        global_state = query.reset_state(query_result, global_state)
      # Expected value is the signal of the current round plus the residual of
      # two continous tree aggregation values. The tree aggregation value can
      # be inferred from the binary representation of the current step.
      expected = scalar_value + tree_node_value * (
          bin(i % frequency + 1)[2:].count('1') -
          bin(i % frequency)[2:].count('1'))
      print(i, query_result, expected)
      self.assertEqual(query_result, expected)


if __name__ == '__main__':
  tf.test.main()

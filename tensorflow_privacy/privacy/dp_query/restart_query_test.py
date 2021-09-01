# Copyright 2021, Google LLC.
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
"""Tests for `restart_query`."""
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import restart_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query


class RestartIndicatorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('zero', 0), ('negative', -1))
  def test_round_raise(self, frequency):
    with self.assertRaisesRegex(
        ValueError, 'Restart frequency should be equal or larger than 1'):
      restart_query.PeriodicRoundRestartIndicator(frequency)

  @parameterized.named_parameters(('f1', 1), ('f2', 2), ('f4', 4), ('f5', 5))
  def test_round_indicator(self, frequency):
    total_steps = 20
    indicator = restart_query.PeriodicRoundRestartIndicator(frequency)
    state = indicator.initialize()
    for i in range(total_steps):
      flag, state = indicator.next(state)
      if i % frequency == frequency - 1:
        self.assertTrue(flag)
      else:
        self.assertFalse(flag)


def _get_l2_clip_fn():

  def l2_clip_fn(record_as_list, clip_value):
    clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_value)
    return clipped_record

  return l2_clip_fn


class RestartQueryTest(tf.test.TestCase, parameterized.TestCase):

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
    indicator = restart_query.PeriodicRoundRestartIndicator(frequency)
    query = tree_aggregation_query.TreeCumulativeSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False)
    query = restart_query.RestartQuery(query, indicator)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
      # Expected value is the combination of cumsum of signal; sum of trees
      # that have been reset; current tree sum. The tree aggregation value can
      # be inferred from the binary representation of the current step.
      expected = (
          scalar_value * (i + 1) +
          i // frequency * tree_node_value * bin(frequency)[2:].count('1') +
          tree_node_value * bin(i % frequency + 1)[2:].count('1'))
      self.assertEqual(query_result, expected)

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
    indicator = restart_query.PeriodicRoundRestartIndicator(frequency)
    query = tree_aggregation_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False)
    query = restart_query.RestartQuery(query, indicator)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state)
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

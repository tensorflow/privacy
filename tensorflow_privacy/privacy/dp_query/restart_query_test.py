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

import unittest

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_privacy.privacy.dp_query import restart_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query


class RoundRestartIndicatorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('zero', 0), ('negative', -1))
  def test_round_raise(self, frequency):
    with self.assertRaisesRegex(
        ValueError, 'Restart frequency should be equal or larger than 1'):
      restart_query.PeriodicRoundRestartIndicator(frequency)

  @parameterized.named_parameters(('zero', 0), ('negative', -1), ('equal', 2),
                                  ('large', 3))
  def test_round_raise_warmup(self, warmup):
    frequency = 2
    with self.assertRaisesRegex(
        ValueError,
        f'Warmup should be between 1 and `frequency-1={frequency-1}`'):
      restart_query.PeriodicRoundRestartIndicator(frequency, warmup)

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

  @parameterized.named_parameters(('f2', 2, 1), ('f4', 4, 3), ('f5', 5, 2))
  def test_round_indicator_warmup(self, frequency, warmup):
    total_steps = 20
    indicator = restart_query.PeriodicRoundRestartIndicator(frequency, warmup)
    state = indicator.initialize()
    for i in range(total_steps):
      flag, state = indicator.next(state)
      if i % frequency == warmup - 1:
        self.assertTrue(flag)
      else:
        self.assertFalse(flag)


class TimeRestartIndicatorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('zero', 0), ('negative', -1.))
  def test_round_raise(self, secs):
    with self.assertRaisesRegex(
        ValueError, 'Restart period_seconds should be larger than 0'):
      restart_query.PeriodicTimeRestartIndicator(secs)

  def test_round_indicator(self):
    indicator = restart_query.PeriodicTimeRestartIndicator(period_seconds=3600 *
                                                           23.5)
    # TODO(b/193679963): use `tf.timestamp` as the default of a member of
    # the `PeriodicTimeRestartIndicator` to unroll the mock test.
    return_time = tf.Variable(
        1627018868.452365)  # 22:41pm PST 5:41am UTC, July 22, initialize
    with unittest.mock.patch.object(
        tf, 'timestamp', return_value=return_time) as mock_func:
      time_stamps = [
          1627022468.452365,  # 23:41pm PST 5:41am UTC, July 22, 1 hr, False
          1627105268.452365,  # 22:41pm PST 5:41am UTC, July 23, 1 day, True
          1627112468.452365,  # 2 hr after restart, False
          1627189508.452365,  # 23.4 hr after restart, False
          1627189904.452365,  # 23.51 hr after restart, True
      ]
      expected_values = [False, True, False, False, True]
      state = indicator.initialize()
      for v, t in zip(expected_values, time_stamps):
        return_time.assign(t)
        mock_func.return_value = return_time
        flag, state = indicator.next(state)
        self.assertEqual(v, flag.numpy())


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
      self.assertEqual(query_result, expected)


if __name__ == '__main__':
  tf.test.main()

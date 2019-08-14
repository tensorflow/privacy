# Copyright 2019, The TensorFlow Authors.
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

"""Tests for QuantileAdaptiveClipSumQuery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.dp_query import quantile_adaptive_clip_sum_query
from privacy.dp_query import test_utils

tf.enable_eager_execution()


class QuantileAdaptiveClipSumQueryTest(tf.test.TestCase):

  def test_sum_no_clip_no_noise(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_clip_no_noise(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=5.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_noise(self):
    record1, record2 = 2.71828, 3.14159
    stddev = 1.0
    clip = 5.0

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=clip,
        noise_multiplier=stddev / clip,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    noised_sums = []
    for _ in xrange(1000):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      noised_sums.append(query_result.numpy())

    result_stddev = np.std(noised_sums)
    self.assertNear(result_stddev, stddev, 0.1)

  def test_average_no_noise(self):
    record1 = tf.constant([5.0, 0.0])   # Clipped to [3.0, 0.0].
    record2 = tf.constant([-1.0, 2.0])  # Not clipped.

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipAverageQuery(
        initial_l2_norm_clip=3.0,
        noise_multiplier=0.0,
        denominator=2.0,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected_average = [1.0, 1.0]
    self.assertAllClose(result, expected_average)

  def test_average_with_noise(self):
    record1, record2 = 2.71828, 3.14159
    sum_stddev = 1.0
    denominator = 2.0
    clip = 3.0

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipAverageQuery(
        initial_l2_norm_clip=clip,
        noise_multiplier=sum_stddev / clip,
        denominator=denominator,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    noised_averages = []
    for _ in range(1000):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      noised_averages.append(query_result.numpy())

    result_stddev = np.std(noised_averages)
    avg_stddev = sum_stddev / denominator
    self.assertNear(result_stddev, avg_stddev, 0.1)

  def test_adaptation_target_zero(self):
    record1 = tf.constant([8.5])
    record2 = tf.constant([-7.25])

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=0.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    global_state = query.initial_global_state()

    initial_clip = global_state.l2_norm_clip
    self.assertAllClose(initial_clip, 10.0)

    # On the first two iterations, nothing is clipped, so the clip goes down
    # by 1.0 (the learning rate). When the clip reaches 8.0, one record is
    # clipped, so the clip goes down by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 7.0).

    expected_sums = [1.25, 1.25, 0.75, 0.25, 0.0]
    expected_clips = [9.0, 8.0, 7.5, 7.0, 7.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      actual_clip = global_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  def test_adaptation_target_one(self):
    record1 = tf.constant([-1.5])
    record2 = tf.constant([2.75])

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=0.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    global_state = query.initial_global_state()

    initial_clip = global_state.l2_norm_clip
    self.assertAllClose(initial_clip, 0.0)

    # On the first two iterations, both are clipped, so the clip goes up
    # by 1.0 (the learning rate). When the clip reaches 2.0, only one record is
    # clipped, so the clip goes up by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 3.0).

    expected_sums = [0.0, 0.0, 0.5, 1.0, 1.25]
    expected_clips = [1.0, 2.0, 2.5, 3.0, 3.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      actual_clip = global_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  def test_adaptation_linspace(self):
    # 100 records equally spaced from 0 to 10 in 0.1 increments.
    # Test that with a decaying learning rate we converge to the correct
    # median with error at most 0.1.
    records = [tf.constant(x) for x in np.linspace(
        0.0, 10.0, num=21, dtype=np.float32)]

    learning_rate = tf.Variable(1.0)

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=0.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=0.5,
        learning_rate=learning_rate,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    global_state = query.initial_global_state()

    for t in range(50):
      tf.assign(learning_rate, 1.0 / np.sqrt(t+1))
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_clip = global_state.l2_norm_clip

      if t > 40:
        self.assertNear(actual_clip, 5.0, 0.25)

  def test_adaptation_all_equal(self):
    # 100 equal records. Test that with a decaying learning rate we converge to
    # that record and bounce around it.
    records = [tf.constant(5.0)] * 20

    learning_rate = tf.Variable(1.0)

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=0.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=0.5,
        learning_rate=learning_rate,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    global_state = query.initial_global_state()

    for t in range(50):
      tf.assign(learning_rate, 1.0 / np.sqrt(t+1))
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_clip = global_state.l2_norm_clip

      if t > 40:
        self.assertNear(actual_clip, 5.0, 0.25)

  def test_ledger(self):
    record1 = tf.constant([8.5])
    record2 = tf.constant([-7.25])

    population_size = tf.Variable(0)
    selection_probability = tf.Variable(1.0)

    query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=1.0,
        target_unclipped_quantile=0.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0)

    query = privacy_ledger.QueryWithLedger(
        query, population_size, selection_probability)

    # First sample.
    tf.assign(population_size, 10)
    tf.assign(selection_probability, 0.1)
    _, global_state = test_utils.run_query(query, [record1, record2])

    expected_queries = [[10.0, 10.0], [0.5, 0.0]]
    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1 = formatted[0]
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sample_1.queries, expected_queries)

    # Second sample.
    tf.assign(population_size, 20)
    tf.assign(selection_probability, 0.2)
    test_utils.run_query(query, [record1, record2], global_state)

    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1, sample_2 = formatted
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sample_1.queries, expected_queries)

    expected_queries_2 = [[9.0, 9.0], [0.5, 0.0]]
    self.assertAllClose(sample_2.population_size, 20.0)
    self.assertAllClose(sample_2.selection_probability, 0.2)
    self.assertAllClose(sample_2.queries, expected_queries_2)


if __name__ == '__main__':
  tf.test.main()

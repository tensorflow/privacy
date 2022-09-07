# Copyright 2022, The TensorFlow Authors.
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
from tensorflow_privacy.privacy.dp_query import distributed_skellam_query
from tensorflow_privacy.privacy.dp_query import test_utils
import tensorflow_probability as tfp


class DistributedSkellamQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_skellam_sum_no_noise(self):
    record1 = tf.constant([2, 0], dtype=tf.int32)
    record2 = tf.constant([-1, 1], dtype=tf.int32)

    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [1, 1]
    self.assertAllClose(query_result, expected)

  def test_skellam_multiple_shapes(self):
    tensor1 = tf.constant([2, 0], dtype=tf.int32)
    tensor2 = tf.constant([-1, 1, 3], dtype=tf.int32)
    record = [tensor1, tensor2]

    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)
    query_result, _ = test_utils.run_query(query, [record, record])
    expected = [2 * tensor1, 2 * tensor2]
    self.assertAllClose(query_result, expected)

  def test_skellam_raise_type_exception(self):
    with self.assertRaises(TypeError):
      record1 = tf.constant([2, 0], dtype=tf.float32)
      record2 = tf.constant([-1, 1], dtype=tf.float32)

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)
      test_utils.run_query(query, [record1, record2])

  def test_skellam_raise_l1_norm_exception(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      record1 = tf.constant([1, 2], dtype=tf.int32)
      record2 = tf.constant([3, 4], dtype=tf.int32)

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=1, l2_norm_bound=100, local_stddev=0.0)
      test_utils.run_query(query, [record1, record2])

  def test_skellam_raise_l2_norm_exception(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      record1 = tf.constant([1, 2], dtype=tf.int32)
      record2 = tf.constant([3, 4], dtype=tf.int32)

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=10, l2_norm_bound=4, local_stddev=0.0)
      test_utils.run_query(query, [record1, record2])

  def test_skellam_sum_with_noise(self):
    """Use only one record to test std."""
    record = tf.constant([1], dtype=tf.int32)
    local_stddev = 1.0

    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10.0, l2_norm_bound=10, local_stddev=local_stddev)

    noised_sums = []
    for _ in range(1000):
      query_result, _ = test_utils.run_query(query, [record])
      noised_sums.append(query_result)

    result_stddev = np.std(noised_sums)
    self.assertNear(result_stddev, local_stddev, 0.1)

  def test_compare_centralized_distributed_skellam(self):
    """Compare the percentiles of distributed and centralized Skellam.

    The test creates a large zero-vector with shape [num_trials, num_users] to
    be processed with the distributed Skellam noise stddev=1. The result is
    summed over the num_users dimension. The centralized result is produced by
    adding noise to a zero vector [num_trials] with stddev = 1*sqrt(num_users).
    Both results are evaluated to match percentiles (25, 50, 75).
    """

    num_trials = 10000
    num_users = 100
    record = tf.zeros([num_trials], dtype=tf.int32)
    local_stddev = 1.0
    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10.0, l2_norm_bound=10, local_stddev=local_stddev)
    distributed_noised = tf.zeros([num_trials], dtype=tf.int32)
    for _ in range(num_users):
      query_result, _ = test_utils.run_query(query, [record])
      distributed_noised += query_result

    def add_noise(v, stddev):
      lam = stddev**2 / 2

      noise_poisson1 = tf.random.poisson(
          lam=lam, shape=tf.shape(v), dtype=v.dtype)
      noise_poisson2 = tf.random.poisson(
          lam=lam, shape=tf.shape(v), dtype=v.dtype)
      res = v + (noise_poisson1 - noise_poisson2)
      return res

    record_centralized = tf.zeros([num_trials], dtype=tf.int32)
    centralized_noised = add_noise(record_centralized,
                                   local_stddev * np.sqrt(num_users))

    tolerance = 5
    self.assertAllClose(
        tfp.stats.percentile(distributed_noised, 50.0),
        tfp.stats.percentile(centralized_noised, 50.0),
        atol=tolerance)
    self.assertAllClose(
        tfp.stats.percentile(distributed_noised, 75.0),
        tfp.stats.percentile(centralized_noised, 75.0),
        atol=tolerance)
    self.assertAllClose(
        tfp.stats.percentile(distributed_noised, 25.0),
        tfp.stats.percentile(centralized_noised, 25.0),
        atol=tolerance)

  def test_skellam_average_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([1, 1], dtype=tf.int32)
      record2 = tf.constant([1, 1], dtype=tf.int32)

      query = distributed_skellam_query.DistributedSkellamAverageQuery(
          l1_norm_bound=3.0,
          l2_norm_bound=3.0,
          local_stddev=0.0,
          denominator=2.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected_average = [1, 1]
      self.assertAllClose(result, expected_average)


if __name__ == '__main__':
  tf.test.main()

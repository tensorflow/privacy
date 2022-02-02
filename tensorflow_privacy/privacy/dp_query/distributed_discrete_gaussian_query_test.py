# Copyright 2021, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import discrete_gaussian_utils
from tensorflow_privacy.privacy.dp_query import distributed_discrete_gaussian_query
from tensorflow_privacy.privacy.dp_query import test_utils

ddg_sum_query = distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery


def silence_tf_error_messages(func):
  """Decorator that temporarily changes the TF logging levels."""

  def wrapper(*args, **kwargs):
    cur_verbosity = tf.compat.v1.logging.get_verbosity()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    func(*args, **kwargs)
    tf.compat.v1.logging.set_verbosity(cur_verbosity)  # Reset verbosity.

  return wrapper


class DistributedDiscreteGaussianQueryTest(tf.test.TestCase,
                                           parameterized.TestCase):

  def test_sum_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([2, 0], dtype=tf.int32)
      record2 = tf.constant([-1, 1], dtype=tf.int32)

      query = ddg_sum_query(l2_norm_bound=10, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1, 1]
      self.assertAllEqual(result, expected)

  @parameterized.product(sample_size=[1, 3])
  def test_sum_multiple_shapes(self, sample_size):
    with self.cached_session() as sess:
      t1 = tf.constant([2, 0], dtype=tf.int32)
      t2 = tf.constant([-1, 1, 3], dtype=tf.int32)
      t3 = tf.constant([-2], dtype=tf.int32)
      record = [t1, t2, t3]
      sample = [record] * sample_size

      query = ddg_sum_query(l2_norm_bound=10, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, sample)
      expected = [sample_size * t1, sample_size * t2, sample_size * t3]
      result, expected = sess.run([query_result, expected])
      # Use `assertAllClose` for nested structures equality (with tolerance=0).
      self.assertAllClose(result, expected, atol=0)

  @parameterized.product(sample_size=[1, 3])
  def test_sum_nested_record_structure(self, sample_size):
    with self.cached_session() as sess:
      t1 = tf.constant([1, 0], dtype=tf.int32)
      t2 = tf.constant([1, 1, 1], dtype=tf.int32)
      t3 = tf.constant([1], dtype=tf.int32)
      t4 = tf.constant([[1, 1], [1, 1]], dtype=tf.int32)
      record = [t1, dict(a=t2, b=[t3, (t4, t1)])]
      sample = [record] * sample_size

      query = ddg_sum_query(l2_norm_bound=10, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, sample)
      result = sess.run(query_result)

      s = sample_size
      expected = [t1 * s, dict(a=t2 * s, b=[t3 * s, (t4 * s, t1 * s)])]
      # Use `assertAllClose` for nested structures equality (with tolerance=0)
      self.assertAllClose(result, expected, atol=0)

  def test_sum_raise_on_float_inputs(self):
    with self.cached_session() as sess:
      record1 = tf.constant([2, 0], dtype=tf.float32)
      record2 = tf.constant([-1, 1], dtype=tf.float32)
      query = ddg_sum_query(l2_norm_bound=10, local_stddev=0.0)

      with self.assertRaises(TypeError):
        query_result, _ = test_utils.run_query(query, [record1, record2])
        sess.run(query_result)

  @parameterized.product(l2_norm_bound=[0, 3, 10, 14.1])
  @silence_tf_error_messages
  def test_sum_raise_on_l2_norm_excess(self, l2_norm_bound):
    with self.cached_session() as sess:
      record = tf.constant([10, 10], dtype=tf.int32)
      query = ddg_sum_query(l2_norm_bound=l2_norm_bound, local_stddev=0.0)

      with self.assertRaises(tf.errors.InvalidArgumentError):
        query_result, _ = test_utils.run_query(query, [record])
        sess.run(query_result)

  def test_sum_float_norm_not_rounded(self):
    """Test that the float L2 norm bound doesn't get rounded/casted to integers."""
    with self.cached_session() as sess:
      # A casted/rounded norm bound would be insufficient.
      l2_norm_bound = 14.2
      record = tf.constant([10, 10], dtype=tf.int32)
      query = ddg_sum_query(l2_norm_bound=l2_norm_bound, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record])
      result = sess.run(query_result)
      expected = [10, 10]
      self.assertAllEqual(result, expected)

  @parameterized.named_parameters([('2_local_stddev_1_record', 2, 1),
                                   ('10_local_stddev_4_records', 10, 4),
                                   ('1000_local_stddev_1_record', 1000, 1),
                                   ('1000_local_stddev_25_records', 1000, 25)])
  def test_sum_local_noise_shares(self, local_stddev, num_records):
    """Test the noise level of the sum of discrete Gaussians applied locally.

    The sum of discrete Gaussians is not a discrete Gaussian, but it will be
    extremely close for sigma >= 2. We will thus compare the aggregated noise
    to a central discrete Gaussian noise with appropriately scaled stddev with
    some reasonable tolerance.

    Args:
      local_stddev: The stddev of the local discrete Gaussian noise.
      num_records: The number of records to be aggregated.
    """
    # Aggregated local noises.
    num_trials = 1000
    record = tf.zeros([num_trials], dtype=tf.int32)
    sample = [record] * num_records
    query = ddg_sum_query(l2_norm_bound=10.0, local_stddev=local_stddev)
    query_result, _ = test_utils.run_query(query, sample)

    # Central discrete Gaussian noise.
    central_stddev = np.sqrt(num_records) * local_stddev
    central_noise = discrete_gaussian_utils.sample_discrete_gaussian(
        scale=tf.cast(tf.round(central_stddev), record.dtype),
        shape=tf.shape(record),
        dtype=record.dtype)

    agg_noise, central_noise = self.evaluate([query_result, central_noise])

    mean_stddev = central_stddev * np.sqrt(num_trials) / num_trials
    atol = 3.5 * mean_stddev

    # Use the atol for mean as a rough default atol for stddev/percentile.
    self.assertAllClose(np.mean(agg_noise), np.mean(central_noise), atol=atol)
    self.assertAllClose(np.std(agg_noise), np.std(central_noise), atol=atol)
    self.assertAllClose(
        np.percentile(agg_noise, [25, 50, 75]),
        np.percentile(central_noise, [25, 50, 75]),
        atol=atol)


if __name__ == '__main__':
  tf.test.main()

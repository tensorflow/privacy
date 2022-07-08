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
"""Implements DPQuery interface for Skellam average queries."""

import collections

import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import normalized_query


class DistributedSkellamSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for discrete distributed sum queries.

  This implementation is for the distributed queries where the Skellam noise
  is applied locally to a discrete vector that matches the norm bound.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l1_norm_bound', 'l2_norm_bound', 'local_stddev'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['l1_norm_bound', 'l2_norm_bound', 'local_stddev'])

  def __init__(self, l1_norm_bound, l2_norm_bound, local_stddev):
    """Initializes the DistributedSkellamSumQuery.

    Args:
      l1_norm_bound: The l1 norm bound to verify for each record.
      l2_norm_bound: The l2 norm bound to verify for each record.
      local_stddev: The standard deviation of the Skellam distribution.
    """
    self._l1_norm_bound = l1_norm_bound
    self._l2_norm_bound = l2_norm_bound
    self._local_stddev = local_stddev

  def initial_global_state(self):
    """Since we operate on discrete values, use int for L1 bound and float for L2 bound."""
    return self._GlobalState(
        tf.cast(self._l1_norm_bound, tf.int32),
        tf.cast(self._l2_norm_bound, tf.float32),
        tf.cast(self._local_stddev, tf.float32))

  def derive_sample_params(self, global_state):
    return self._SampleParams(global_state.l1_norm_bound,
                              global_state.l2_norm_bound,
                              global_state.local_stddev)

  def add_noise_to_sample(self, local_stddev, record):
    """Adds Skellam noise to the sample.

    We use difference of two Poisson random variable with lambda hyperparameter
    that equals 'local_stddev**2/2' that results in a standard deviation
    'local_stddev' for the Skellam noise to be added locally.

    Args:
      local_stddev: The standard deviation of the local Skellam noise.
      record: The record to be processed.

    Returns:
      A record with added noise.
    """
    # Use float64 as the stddev could be large after quantization.
    local_stddev = tf.cast(local_stddev, tf.float64)
    poisson_lam = 0.5 * local_stddev * local_stddev

    def add_noise(v):
      poissons = tf.random.stateless_poisson(
          shape=tf.concat([tf.shape(v), [2]], axis=0),  # Two draws of Poisson.
          seed=tf.cast([tf.timestamp() * 10**6, 0], tf.int64),
          lam=[poisson_lam, poisson_lam],
          dtype=tf.int64)
      return v + tf.cast(poissons[..., 0] - poissons[..., 1], v.dtype)

    return tf.nest.map_structure(add_noise, record)

  def preprocess_record(self, params, record):
    """Check record norm and add noise to the record.

    For both L1 and L2 norms we compute a global norm of the provided record.
    Since the record contains int32 tensors we cast them into float32 to
    compute L2 norm. In the end we run three asserts: type, l1, and l2 norms.

    Args:
      params: The parameters for the particular sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, params) where `preprocessed_records` is
        the structure of preprocessed tensors, and params contains sample
        params.
    """
    record_as_list = tf.nest.flatten(record)
    record_as_float = [tf.cast(x, tf.float32) for x in record_as_list]
    tf.nest.map_structure(lambda x: tf.debugging.assert_type(x, tf.int32),
                          record_as_list)
    dependencies = [
        tf.debugging.assert_less_equal(
            tf.reduce_sum([tf.norm(x, ord=1) for x in record_as_list]),
            params.l1_norm_bound,
            message=f'L1 norm exceeds {params.l1_norm_bound}.'),
        tf.debugging.assert_less_equal(
            tf.linalg.global_norm(record_as_float),
            params.l2_norm_bound,
            message=f'Global L2 norm exceeds {params.l2_norm_bound}.')
    ]
    with tf.control_dependencies(dependencies):
      record = tf.cond(
          tf.equal(params.local_stddev, 0), lambda: record,
          lambda: self.add_noise_to_sample(params.local_stddev, record))
      return record

  def get_noised_result(self, sample_state, global_state):
    """The noise was already added locally, therefore just continue."""
    event = dp_accounting.UnsupportedDpEvent()
    return sample_state, global_state, event


class DistributedSkellamAverageQuery(normalized_query.NormalizedQuery):
  """Implements DPQuery interface for Skellam average queries.

  Checks norm bounds and adds Skellam noise to each vector, sums them up, casts
  to float32 and normalizes using the truediv operation.
  """

  def __init__(self, l1_norm_bound, l2_norm_bound, local_stddev, denominator):
    """Initializes the GaussianAverageQuery.

    Args:
      l1_norm_bound: The l1 norm bound to verify for each record.
      l2_norm_bound: The l2 norm bound to verify for each record.
      local_stddev: The local_stddev of the noise added to each record (before
        sum and normalization).
      denominator: The normalization constant (applied after sum).
    """
    super().__init__(
        numerator_query=DistributedSkellamSumQuery(l1_norm_bound, l2_norm_bound,
                                                   local_stddev),
        denominator=denominator)

  def get_noised_result(self, sample_state, global_state):
    """Normalize accumulated sum with floordiv."""
    noised_sum, new_sum_global_state, event = self._numerator.get_noised_result(
        sample_state, global_state.numerator_state)

    def normalize(v):
      return tf.math.truediv(tf.cast(v, tf.float32), global_state.denominator)

    return (tf.nest.map_structure(normalize, noised_sum),
            self._GlobalState(new_sum_global_state,
                              global_state.denominator), event)

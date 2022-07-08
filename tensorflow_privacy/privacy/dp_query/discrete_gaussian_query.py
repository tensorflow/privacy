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
"""Implements DPQuery interface for discrete Gaussian mechanism."""

import collections

import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import discrete_gaussian_utils
from tensorflow_privacy.privacy.dp_query import dp_query


class DiscreteGaussianSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery for discrete Gaussian sum queries.

  For each local record, we check the L2 norm bound and add discrete Gaussian
  noise. In particular, this DPQuery does not perform L2 norm clipping and the
  norms of the input records are expected to be bounded.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_bound', 'stddev'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple('_SampleParams',
                                         ['l2_norm_bound', 'stddev'])

  def __init__(self, l2_norm_bound, stddev):
    """Initializes the DiscreteGaussianSumQuery.

    Args:
      l2_norm_bound: The L2 norm bound to verify for each record.
      stddev: The stddev of the discrete Gaussian noise added to the sum.
    """
    self._l2_norm_bound = l2_norm_bound
    self._stddev = stddev

  def initial_global_state(self):
    return self._GlobalState(
        tf.cast(self._l2_norm_bound, tf.float32),
        tf.cast(self._stddev, tf.float32))

  def derive_sample_params(self, global_state):
    return self._SampleParams(global_state.l2_norm_bound, global_state.stddev)

  def preprocess_record(self, params, record):
    """Check record norm and add noise to the record."""
    record_as_list = tf.nest.flatten(record)
    record_as_float_list = [tf.cast(x, tf.float32) for x in record_as_list]
    tf.nest.map_structure(lambda x: tf.compat.v1.assert_type(x, tf.int32),
                          record_as_list)
    dependencies = [
        tf.compat.v1.assert_less_equal(
            tf.linalg.global_norm(record_as_float_list),
            params.l2_norm_bound,
            message=f'Global L2 norm exceeds {params.l2_norm_bound}.')
    ]
    with tf.control_dependencies(dependencies):
      return tf.nest.map_structure(tf.identity, record)

  def get_noised_result(self, sample_state, global_state):
    """Adds discrete Gaussian noise to the aggregate."""
    # Round up the noise as the TF discrete Gaussian sampler only takes
    # integer noise stddevs for now.
    ceil_stddev = tf.cast(tf.math.ceil(global_state.stddev), tf.int32)

    def add_noise(v):
      noised_v = v + discrete_gaussian_utils.sample_discrete_gaussian(
          scale=ceil_stddev, shape=tf.shape(v), dtype=v.dtype)
      # Ensure shape as TF shape inference may fail due to custom noise sampler.
      return tf.ensure_shape(noised_v, v.shape)

    result = tf.nest.map_structure(add_noise, sample_state)
    event = dp_accounting.UnsupportedDpEvent()
    return result, global_state, event

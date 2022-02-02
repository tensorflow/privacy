# Copyright 2020, The TensorFlow Authors.
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
"""Implements DPQuery interface for normalized queries."""

import collections

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query


class NormalizedQuery(dp_query.SumAggregationDPQuery):
  """`DPQuery` for queries with a `DPQuery` numerator and fixed denominator.

  If the number of records per round is a public constant R, `NormalizedQuery`
  could be used with a sum query as the numerator and R as the denominator to
  implement an average. Under some sampling schemes, such as Poisson
  subsampling, the actual number of records in a sample is a private quantity,
  so we cannot use it directly. Using this class with the expected number of
  records as the denominator gives an unbiased estimate of the average.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['numerator_state', 'denominator'])

  def __init__(self, numerator_query, denominator):
    """Initializes the NormalizedQuery.

    Args:
      numerator_query: A SumAggregationDPQuery for the numerator.
      denominator: A value for the denominator. May be None if it will be
        supplied via the set_denominator function before get_noised_result is
        called.
    """
    self._numerator = numerator_query
    self._denominator = denominator

    assert isinstance(self._numerator, dp_query.SumAggregationDPQuery)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    denominator = tf.cast(self._denominator, tf.float32)
    return self._GlobalState(self._numerator.initial_global_state(),
                             denominator)

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._numerator.derive_sample_params(global_state.numerator_state)

  def initial_sample_state(self, template):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    # NormalizedQuery has no sample state beyond the numerator state.
    return self._numerator.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    return self._numerator.preprocess_record(params, record)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_sum, new_sum_global_state, event = self._numerator.get_noised_result(
        sample_state, global_state.numerator_state)

    def normalize(v):
      return tf.truediv(v, global_state.denominator)

    # The denominator is constant so the privacy cost comes from the numerator.
    return (tf.nest.map_structure(normalize, noised_sum),
            self._GlobalState(new_sum_global_state,
                              global_state.denominator), event)

  def derive_metrics(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`."""
    return self._numerator.derive_metrics(global_state.numerator_state)

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
"""Implements DPQuery interface for no privacy average queries."""

import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query


class NoPrivacySumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for a sum query with no privacy.

  Accumulates vectors without clipping or adding noise.
  """

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    return sample_state, global_state, dp_accounting.NonPrivateDpEvent()


class NoPrivacyAverageQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for an average query with no privacy.

  Accumulates vectors and normalizes by the total number of accumulated vectors.
  Under some sampling schemes, such as Poisson subsampling, the number of
  records in a sample is a private quantity, so we lose all privacy guarantees
  by using the number of records directly to normalize.

  Also allows weighted accumulation, unlike the base class DPQuery. In a private
  implementation of weighted average, the weight would have to be itself
  privatized.
  """

  def initial_sample_state(self, template):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return super().initial_sample_state(template), tf.constant(0.0)

  def preprocess_record(self, params, record, weight=1):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Optional `weight` argument allows weighted accumulation.

    Args:
      params: The parameters for the sample.
      record: The record to accumulate.
      weight: Optional weight for the record.

    Returns:
      The preprocessed record.
    """
    weighted_record = tf.nest.map_structure(lambda t: weight * t, record)
    return (weighted_record, tf.cast(weight, tf.float32))

  def accumulate_record(self, params, sample_state, record, weight=1):
    """Implements `tensorflow_privacy.DPQuery.accumulate_record`.

    Optional `weight` argument allows weighted accumulation.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.
      weight: Optional weight for the record.

    Returns:
      The updated sample state.
    """
    weighted_record = tf.nest.map_structure(lambda t: weight * t, record)
    return self.accumulate_preprocessed_record(
        sample_state, (weighted_record, tf.cast(weight, tf.float32)))

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    sum_state, denominator = sample_state

    result = tf.nest.map_structure(lambda t: t / denominator, sum_state)
    return result, global_state, dp_accounting.NonPrivateDpEvent()

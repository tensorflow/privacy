# Copyright 2018, The TensorFlow Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
import tensorflow as tf

from privacy.dp_query import dp_query

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  nest = tf.contrib.framework.nest
else:
  nest = tf.nest


class NoPrivacySumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for a sum query with no privacy.

  Accumulates vectors without clipping or adding noise.
  """

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    return sample_state, global_state


class NoPrivacyAverageQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for an average query with no privacy.

  Accumulates vectors and normalizes by the total number of accumulated vectors.
  """

  def initial_sample_state(self, global_state, template):
    """See base class."""
    return (
        super(NoPrivacyAverageQuery, self).initial_sample_state(
            global_state, template),
        tf.constant(0.0))

  def preprocess_record(self, params, record, weight=1):
    """Multiplies record by weight."""
    weighted_record = nest.map_structure(lambda t: weight * t, record)
    return (weighted_record, tf.cast(weight, tf.float32))

  def accumulate_record(self, params, sample_state, record, weight=1):
    """Accumulates record, multiplying by weight."""
    weighted_record = nest.map_structure(lambda t: weight * t, record)
    return self.accumulate_preprocessed_record(
        sample_state, (weighted_record, tf.cast(weight, tf.float32)))

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    sum_state, denominator = sample_state

    return (
        nest.map_structure(lambda t: t / denominator, sum_state),
        global_state)

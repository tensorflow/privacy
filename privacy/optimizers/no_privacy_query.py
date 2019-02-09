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

import tensorflow as tf

from privacy.optimizers import dp_query

nest = tf.contrib.framework.nest


class NoPrivacySumQuery(dp_query.DPQuery):
  """Implements DPQuery interface for a sum query with no privacy.

  Accumulates vectors without clipping or adding noise.
  """

  def initial_global_state(self):
    """Returns the initial global state for the NoPrivacySumQuery."""
    return None

  def derive_sample_params(self, global_state):
    """See base class."""
    del global_state  # unused.
    return None

  def initial_sample_state(self, global_state, tensors):
    """See base class."""
    del global_state  # unused.
    return nest.map_structure(tf.zeros_like, tensors)

  def accumulate_record(self, params, sample_state, record, weight=1):
    """See base class. Optional argument for weighted sum queries."""
    del params  # unused.

    def add_weighted(state_tensor, record_tensor):
      return tf.add(state_tensor, weight * record_tensor)

    return nest.map_structure(add_weighted, sample_state, record)

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    return sample_state, global_state


class NoPrivacyAverageQuery(dp_query.DPQuery):
  """Implements DPQuery interface for an average query with no privacy.

  Accumulates vectors and normalizes by the total number of accumulated vectors.
  """

  def __init__(self):
    """Initializes the NoPrivacyAverageQuery."""
    self._numerator = NoPrivacySumQuery()

  def initial_global_state(self):
    """Returns the initial global state for the NoPrivacyAverageQuery."""
    return self._numerator.initial_global_state()

  def derive_sample_params(self, global_state):
    """See base class."""
    del global_state  # unused.
    return None

  def initial_sample_state(self, global_state, tensors):
    """See base class."""
    return self._numerator.initial_sample_state(global_state, tensors), 0.0

  def accumulate_record(self, params, sample_state, record, weight=1):
    """See base class. Optional argument for weighted average queries."""
    sum_sample_state, denominator = sample_state
    return (
        self._numerator.accumulate_record(
            params, sum_sample_state, record, weight),
        tf.add(denominator, weight))

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    sum_sample_state, denominator = sample_state
    exact_sum, new_global_state = self._numerator.get_noised_result(
        sum_sample_state, global_state)

    def normalize(v):
      return tf.truediv(v, denominator)

    return nest.map_structure(normalize, exact_sum), new_global_state

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

"""Implements DPQuery interface for normalized queries.
"""

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


class NormalizedQuery(dp_query.DPQuery):
  """DPQuery for queries with a DPQuery numerator and fixed denominator."""

  def __init__(self, numerator_query, denominator):
    """Initializer for NormalizedQuery.

    Args:
      numerator_query: A DPQuery for the numerator.
      denominator: A value for the denominator.
    """
    self._numerator = numerator_query
    self._denominator = tf.cast(denominator,
                                tf.float32) if denominator is not None else None

  def initial_global_state(self):
    """Returns the initial global state for the NormalizedQuery."""
    # NormalizedQuery has no global state beyond the numerator state.
    return self._numerator.initial_global_state()

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    return self._numerator.derive_sample_params(global_state)

  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    # NormalizedQuery has no sample state beyond the numerator state.
    return self._numerator.initial_sample_state(global_state, tensors)

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      The updated sample state.
    """
    return self._numerator.accumulate_record(params, sample_state, record)

  def get_noised_result(self, sample_state, global_state):
    """Gets noised average after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (estimate, new_global_state) where "estimate" is the estimated
      average of the records and "new_global_state" is the updated global state.
    """
    noised_sum, new_sum_global_state = self._numerator.get_noised_result(
        sample_state, global_state)
    def normalize(v):
      return tf.truediv(v, self._denominator)

    return nest.map_structure(normalize, noised_sum), new_sum_global_state

  def set_denominator(self, denominator):
    self._denominator = tf.cast(denominator, tf.float32)

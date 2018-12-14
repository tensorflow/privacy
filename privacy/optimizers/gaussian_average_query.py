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

"""Implements PrivateQuery interface for Gaussian average queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from privacy.optimizers import private_queries


class GaussianAverageQuery(private_queries.PrivateAverageQuery):
  """Implements PrivateQuery interface for Gaussian average queries.

  Accumulates clipped vectors, then adds Gaussian noise to the average.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev', 'denominator'])

  def __init__(self, l2_norm_clip, stddev, denominator):
    """Initializes the GaussianAverageQuery."""
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev
    self._denominator = denominator

  def initial_global_state(self):
    """Returns the initial global state for the PrivacyHelper."""
    return self._GlobalState(
        float(self._l2_norm_clip), float(self._stddev),
        float(self._denominator))

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    return global_state.l2_norm_clip

  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    del global_state  # unused.
    return tf.contrib.framework.nest.map_structure(tf.zeros_like, tensors)

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      The updated sample state.
    """
    l2_norm_clip = params
    clipped, _ = tf.clip_by_global_norm(record, l2_norm_clip)
    return tf.contrib.framework.nest.map_structure(tf.add, sample_state,
                                                   clipped)

  def get_noised_average(self, sample_state, global_state):
    """Gets noised average after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (estimate, new_global_state) where "estimate" is the estimated
      average of the records and "new_global_state" is the updated global state.
    """
    def noised_average(v):
      return tf.truediv(
          v + tf.random_normal(tf.shape(v), stddev=self._stddev),
          global_state.denominator)

    return (tf.contrib.framework.nest.map_structure(noised_average,
                                                    sample_state), global_state)

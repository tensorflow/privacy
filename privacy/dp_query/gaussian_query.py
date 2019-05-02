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

"""Implements DPQuery interface for Gaussian average queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
import tensorflow as tf

from privacy.dp_query import dp_query
from privacy.dp_query import normalized_query

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  nest = tf.contrib.framework.nest
else:
  nest = tf.nest


class GaussianSumQuery(dp_query.DPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Accumulates clipped vectors, then adds Gaussian noise to the sum.
  """

  def __init__(self, l2_norm_clip, stddev, ledger=None):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
      ledger: The privacy ledger to which queries should be recorded.
    """
    self._l2_norm_clip = tf.cast(l2_norm_clip, tf.float32)
    self._stddev = tf.cast(stddev, tf.float32)
    self._ledger = ledger

  def initial_global_state(self):
    """Returns the initial global state for the GaussianSumQuery."""
    return None

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    return self._l2_norm_clip

  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    if self._ledger:
      dependencies = [
          self._ledger.record_sum_query(self._l2_norm_clip, self._stddev)
      ]
    else:
      dependencies = []
    with tf.control_dependencies(dependencies):
      return nest.map_structure(tf.zeros_like, tensors)

  def accumulate_record_impl(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      A tuple containing the updated sample state and the global norm.
    """
    l2_norm_clip = params
    record_as_list = nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    clipped = nest.pack_sequence_as(record, clipped_as_list)
    return nest.map_structure(tf.add, sample_state, clipped), norm

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      The updated sample state.
    """
    new_sample_state, _ = self.accumulate_record_impl(
        params, sample_state, record)
    return new_sample_state

  def get_noised_result(self, sample_state, global_state):
    """Gets noised sum after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (estimate, new_global_state) where "estimate" is the estimated
      sum of the records and "new_global_state" is the updated global state.
    """
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
      def add_noise(v):
        return v + tf.random_normal(tf.shape(v), stddev=self._stddev)
    else:
      random_normal = tf.random_normal_initializer(stddev=self._stddev)
      def add_noise(v):
        return v + random_normal(tf.shape(v))

    return nest.map_structure(add_noise, sample_state), global_state


class GaussianAverageQuery(normalized_query.NormalizedQuery):
  """Implements DPQuery interface for Gaussian average queries.

  Accumulates clipped vectors, adds Gaussian noise, and normalizes.

  Note that we use "fixed-denominator" estimation: the denominator should be
  specified as the expected number of records per sample. Accumulating the
  denominator separately would also be possible but would be produce a higher
  variance estimator.
  """

  def __init__(self,
               l2_norm_clip,
               sum_stddev,
               denominator,
               ledger=None):
    """Initializes the GaussianAverageQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      sum_stddev: The stddev of the noise added to the sum (before
        normalization).
      denominator: The normalization constant (applied after noise is added to
        the sum).
      ledger: The privacy ledger to which queries should be recorded.
    """
    super(GaussianAverageQuery, self).__init__(
        numerator_query=GaussianSumQuery(l2_norm_clip, sum_stddev, ledger),
        denominator=denominator)

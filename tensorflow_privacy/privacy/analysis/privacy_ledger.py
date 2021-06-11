# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PrivacyLedger class for keeping a record of private queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import tensor_buffer
from tensorflow_privacy.privacy.dp_query import dp_query

SampleEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'SampleEntry', ['population_size', 'selection_probability', 'queries'])

GaussianSumQueryEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'GaussianSumQueryEntry', ['l2_norm_bound', 'noise_stddev'])


def format_ledger(sample_array, query_array):
  """Converts array representation into a list of SampleEntries."""
  samples = []
  query_pos = 0
  sample_pos = 0
  for sample in sample_array:
    population_size, selection_probability, num_queries = sample
    queries = []
    for _ in range(int(num_queries)):
      query = query_array[query_pos]
      assert int(query[0]) == sample_pos
      queries.append(GaussianSumQueryEntry(*query[1:]))
      query_pos += 1
    samples.append(SampleEntry(population_size, selection_probability, queries))
    sample_pos += 1
  return samples


class PrivacyLedger(object):
  """Class for keeping a record of private queries.

  The PrivacyLedger keeps a record of all queries executed over a given dataset
  for the purpose of computing privacy guarantees. To use it, it must be
  associated with a `DPQuery` object via a `QueryWithLedger`.

  The current implementation works only with DPQueries that consist of composing
  Gaussian sum mechanism with Poisson subsampling.

  Example usage:

  ```
  import tensorflow_privacy as tfp

  dp_query = tfp.QueryWithLedger(
    tensorflow_privacy.GaussianSumQuery(
      l2_norm_clip=1.0, stddev=1.0),
    population_size=10000,
    selection_probability=0.01)

  # Use dp_query here in training loop.

  formatted_ledger = dp_query.ledger.get_formatted_ledger_eager()
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
          list(range(5, 64)) + [128, 256, 512])
  total_rdp = tfp.compute_rdp_from_ledger(formatted_ledger, orders)
  epsilon = tfp.get_privacy_spent(orders, total_rdp, target_delta=1e-5)
  ```
  """

  def __init__(self,
               population_size,
               selection_probability):
    """Initializes the PrivacyLedger.

    Args:
      population_size: An integer (may be variable) specifying the size of the
        population, i.e. size of the training data used in each epoch.
      selection_probability: A floating point value (may be variable) specifying
        the probability each record is included in a sample.

    Raises:
      ValueError: If `selection_probability` is 0.
    """
    self._population_size = population_size
    self._selection_probability = selection_probability

    if tf.executing_eagerly():
      if tf.equal(selection_probability, 0):
        raise ValueError('Selection probability cannot be 0.')
      init_capacity = tf.cast(tf.math.ceil(1 / selection_probability), tf.int32)
    else:
      if selection_probability == 0:
        raise ValueError('Selection probability cannot be 0.')
      init_capacity = np.int(np.ceil(1 / selection_probability))

    # The query buffer stores rows corresponding to GaussianSumQueryEntries.
    self._query_buffer = tensor_buffer.TensorBuffer(
        init_capacity, [3], tf.float32, 'query')
    self._sample_var = tf.Variable(
        initial_value=tf.zeros([3]), trainable=False, name='sample')

    # The sample buffer stores rows corresponding to SampleEntries.
    self._sample_buffer = tensor_buffer.TensorBuffer(
        init_capacity, [3], tf.float32, 'sample')
    self._sample_count = tf.Variable(
        initial_value=0.0, trainable=False, name='sample_count')
    self._query_count = tf.Variable(
        initial_value=0.0, trainable=False, name='query_count')
    self._cs = tf.CriticalSection()

  def record_sum_query(self, l2_norm_bound, noise_stddev):
    """Records that a query was issued.

    Args:
      l2_norm_bound: The maximum l2 norm of the tensor group in the query.
      noise_stddev: The standard deviation of the noise applied to the sum.

    Returns:
      An operation recording the sum query to the ledger. This should be called
      for every Gaussian sum query that is issued on a sample.
    """

    def _do_record_query():
      with tf.control_dependencies(
          [tf.assign(self._query_count, self._query_count + 1)]):
        return self._query_buffer.append(
            [self._sample_count, l2_norm_bound, noise_stddev])

    return self._cs.execute(_do_record_query)

  def finalize_sample(self):
    """Finalizes sample and records sample ledger entry.

    This should be called once per application of the mechanism on a sample,
    after all sum queries have been recorded.

    Returns:
      An operation recording the complete mechanism (sampling and sum
      estimation) to the ledger.
    """
    with tf.control_dependencies([
        tf.assign(self._sample_var, [
            self._population_size, self._selection_probability,
            self._query_count
        ])
    ]):
      with tf.control_dependencies([
          tf.assign(self._sample_count, self._sample_count + 1),
          tf.assign(self._query_count, 0)
      ]):
        return self._sample_buffer.append(self._sample_var)

  def get_unformatted_ledger(self):
    """Returns the raw sample and query values."""
    return self._sample_buffer.values, self._query_buffer.values

  def get_formatted_ledger(self, sess):
    """Gets the formatted query ledger.

    Args:
      sess: The tensorflow session in which the ledger was created.

    Returns:
      The query ledger as a list of `SampleEntry` instances.
    """
    sample_array = sess.run(self._sample_buffer.values)
    query_array = sess.run(self._query_buffer.values)

    return format_ledger(sample_array, query_array)

  def get_formatted_ledger_eager(self):
    """Gets the formatted query ledger.

    Returns:
      The query ledger as a list of `SampleEntry` instances.
    """
    sample_array = self._sample_buffer.values.numpy()
    query_array = self._query_buffer.values.numpy()

    return format_ledger(sample_array, query_array)


class QueryWithLedger(dp_query.DPQuery):
  """A class for DP queries that record events to a `PrivacyLedger`.

  `QueryWithLedger` should be the top-level query in a structure of queries that
  may include sum queries, nested queries, etc. It should simply wrap another
  query and contain a reference to the ledger. Any contained queries (including
  those contained in the leaves of a nested query) should also contain a
  reference to the same ledger object.

  Only composed Gaussian sum queries with Poisson subsampling are supported.
  This includes `GaussianSumQuery`, `QuantileEstimatorQuery`, and
  `QuantileAdaptiveClipSumQuery`, as well as `NestedQuery` or `NormalizedQuery`
  objects that contain the previous mentioned query types.
  """

  def __init__(self, query,
               population_size=None, selection_probability=None,
               ledger=None):
    """Initializes the `QueryWithLedger`.

    Args:
      query: The query whose events should be recorded to the ledger. Any
        subqueries (including those in the leaves of a nested query) should also
        contain a reference to the same ledger given here.
      population_size: An integer (may be variable) specifying the size of the
        population, i.e. size of the training data used in each epoch. May be
        `None` if `ledger` is specified.
      selection_probability: A floating point value (may be variable) specifying
        the probability each record is included in a sample under Poisson
        subsampling. May be `None` if `ledger` is specified.
      ledger: A `PrivacyLedger` to use. Must be specified if either of
        `population_size` or `selection_probability` is `None`.
    """
    self._query = query
    if population_size is not None and selection_probability is not None:
      self.set_ledger(PrivacyLedger(population_size, selection_probability))
    elif ledger is not None:
      self.set_ledger(ledger)
    else:
      raise ValueError('One of (population_size, selection_probability) or '
                       'ledger must be specified.')

  @property
  def ledger(self):
    """Gets the ledger that all inner queries record to."""
    return self._ledger

  def set_ledger(self, ledger):
    """Sets a new ledger."""
    self._ledger = ledger
    self._query.set_ledger(ledger)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._query.initial_global_state()

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._query.derive_sample_params(global_state)

  def initial_sample_state(self, template):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._query.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    return self._query.preprocess_record(params, record)

  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """Implements `tensorflow_privacy.DPQuery.accumulate_preprocessed_record`."""
    return self._query.accumulate_preprocessed_record(
        sample_state, preprocessed_record)

  def merge_sample_states(self, sample_state_1, sample_state_2):
    """Implements `tensorflow_privacy.DPQuery.merge_sample_states`."""
    return self._query.merge_sample_states(sample_state_1, sample_state_2)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`.

    Besides noising and returning the result of the inner query, ensures that
    the sample is recorded to the ledger.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state, storing long-term privacy bookkeeping.

    Returns:
      A tuple (result, new_global_state) where "result" is the result of the
      query and "new_global_state" is the updated global state.
    """
    # Ensure sample_state is fully aggregated before calling get_noised_result.
    with tf.control_dependencies(tf.nest.flatten(sample_state)):
      result, new_global_state = self._query.get_noised_result(
          sample_state, global_state)

    # Ensure inner queries have recorded before finalizing.
    with tf.control_dependencies(tf.nest.flatten(result)):
      finalize = self._ledger.finalize_sample()

    # Ensure finalizing happens.
    with tf.control_dependencies([finalize]):
      return tf.nest.map_structure(tf.identity, result), new_global_state

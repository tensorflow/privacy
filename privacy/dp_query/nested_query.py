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

"""Implements DPQuery interface for queries over nested structures.
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


class NestedQuery(dp_query.DPQuery):
  """Implements DPQuery interface for structured queries.

  NestedQuery evaluates arbitrary nested structures of queries. Records must be
  nested structures of tensors that are compatible (in type and arity) with the
  query structure, but are allowed to have deeper structure within each leaf of
  the query structure. For example, the nested query [q1, q2] is compatible with
  the record [t1, t2] or [t1, (t2, t3)], but not with (t1, t2), [t1] or
  [t1, t2, t3]. The entire substructure of each record corresponding to a leaf
  node of the query structure is routed to the corresponding query. If the same
  tensor should be consumed by multiple sub-queries, it can be replicated in the
  record, for example [t1, t1].

  NestedQuery is intended to allow privacy mechanisms for groups as described in
  [McMahan & Andrew, 2018: "A General Approach to Adding Differential Privacy to
  Iterative Training Procedures" (https://arxiv.org/abs/1812.06210)].
  """

  def __init__(self, queries):
    """Initializes the NestedQuery.

    Args:
      queries: A nested structure of queries.
    """
    self._queries = queries

  def _map_to_queries(self, fn, *inputs, **kwargs):
    def caller(query, *args):
      return getattr(query, fn)(*args, **kwargs)
    return nest.map_structure_up_to(
        self._queries, caller, self._queries, *inputs)

  def set_ledger(self, ledger):
    self._map_to_queries('set_ledger', ledger=ledger)

  def initial_global_state(self):
    """See base class."""
    return self._map_to_queries('initial_global_state')

  def derive_sample_params(self, global_state):
    """See base class."""
    return self._map_to_queries('derive_sample_params', global_state)

  def initial_sample_state(self, template):
    """See base class."""
    return self._map_to_queries('initial_sample_state', template)

  def preprocess_record(self, params, record):
    """See base class."""
    return self._map_to_queries('preprocess_record', params, record)

  def accumulate_preprocessed_record(
      self, sample_state, preprocessed_record):
    """See base class."""
    return self._map_to_queries(
        'accumulate_preprocessed_record',
        sample_state,
        preprocessed_record)

  def merge_sample_states(self, sample_state_1, sample_state_2):
    return self._map_to_queries(
        'merge_sample_states', sample_state_1, sample_state_2)

  def get_noised_result(self, sample_state, global_state):
    """Gets query result after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (result, new_global_state) where "result" is a structure matching
      the query structure containing the results of the subqueries and
      "new_global_state" is a structure containing the updated global states
      for the subqueries.
    """
    estimates_and_new_global_states = self._map_to_queries(
        'get_noised_result', sample_state, global_state)

    flat_estimates, flat_new_global_states = zip(
        *nest.flatten_up_to(self._queries, estimates_and_new_global_states))
    return (
        nest.pack_sequence_as(self._queries, flat_estimates),
        nest.pack_sequence_as(self._queries, flat_new_global_states))

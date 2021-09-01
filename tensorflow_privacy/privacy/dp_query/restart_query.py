# Copyright 2021, The TensorFlow Authors.
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
"""Implements DPQuery interface for restarting the states of another query.

This query is used to compose with a DPQuery that has `reset_state` function.
"""
import abc
import collections

import tensorflow as tf

from tensorflow_privacy.privacy.dp_query import dp_query


class RestartIndicator(metaclass=abc.ABCMeta):
  """Base class establishing interface for restarting the tree state.

  A `RestartIndicator` maintains a state, and each time `next` is called, a bool
  value is generated to indicate whether to restart, and the indicator state is
  advanced.
  """

  @abc.abstractmethod
  def initialize(self):
    """Makes an initialized state for `RestartIndicator`.

    Returns:
      An initial state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def next(self, state):
    """Gets next bool indicator and advances the `RestartIndicator` state.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is bool indicator and new_state
        is the advanced state.
    """
    raise NotImplementedError


class PeriodicRoundRestartIndicator(RestartIndicator):
  """Indicator for resetting the tree state after every a few number of queries.

  The indicator will maintain an internal counter as state.
  """

  def __init__(self, frequency: int):
    """Construct the `PeriodicRoundRestartIndicator`.

    Args:
      frequency: The `next` function will return `True` every `frequency` number
        of `next` calls.
    """
    if frequency < 1:
      raise ValueError('Restart frequency should be equal or larger than 1 '
                       f'got {frequency}')
    self.frequency = tf.constant(frequency, tf.int32)

  def initialize(self):
    """Returns initialized state of 0 for `PeriodicRoundRestartIndicator`."""
    return tf.constant(0, tf.int32)

  def next(self, state):
    """Gets next bool indicator and advances the state.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is the bool indicator and new_state
        of `state+1`.
    """
    state = state + tf.constant(1, tf.int32)
    flag = state % self.frequency == 0
    return flag, state


class RestartQuery(dp_query.SumAggregationDPQuery):
  """`DPQuery` for `SumAggregationDPQuery` with a `reset_state` function."""

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['inner_query_state', 'indicator_state'])

  def __init__(self, inner_query: dp_query.SumAggregationDPQuery,
               restart_indicator: RestartIndicator):
    """Initializes `RestartQuery`.

    Args:
      inner_query: A `SumAggregationDPQuery` has `reset_state` attribute.
      restart_indicator: A `RestartIndicator` to generate the boolean indicator
        for resetting the state.
    """
    if not hasattr(inner_query, 'reset_state'):
      raise ValueError(f'{type(inner_query)} must define `reset_state` to be '
                       'composed with `RestartQuery`.')
    self._inner_query = inner_query
    self._restart_indicator = restart_indicator

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._GlobalState(
        inner_query_state=self._inner_query.initial_global_state(),
        indicator_state=self._restart_indicator.initialize())

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._inner_query.derive_sample_params(
        global_state.inner_query_state)

  def initial_sample_state(self, template):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._inner_query.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    return self._inner_query.preprocess_record(params, record)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_results, inner_state, event = self._inner_query.get_noised_result(
        sample_state, global_state.inner_query_state)
    restart_flag, indicator_state = self._restart_indicator.next(
        global_state.indicator_state)
    if restart_flag:
      inner_state = self._inner_query.reset_state(noised_results, inner_state)
    return (noised_results, self._GlobalState(inner_state,
                                              indicator_state), event)

  def derive_metrics(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`."""
    return self._inner_query.derive_metrics(global_state.inner_query_state)

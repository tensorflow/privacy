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
"""`DPQuery` for Gaussian sum queries with adaptive clipping."""

import collections

import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import quantile_estimator_query


class QuantileAdaptiveClipSumQuery(dp_query.SumAggregationDPQuery):
  """`DPQuery` for Gaussian sum queries with adaptive clipping.

  Clipping norm is tuned adaptively to converge to a value such that a specified
  quantile of updates are clipped, using the algorithm of Andrew et al. (
  https://arxiv.org/abs/1905.03871). See the paper for details and suggested
  hyperparameter settings.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState',
      ['noise_multiplier', 'sum_state', 'quantile_estimator_state'])

  # pylint: disable=invalid-name
  _SampleState = collections.namedtuple(
      '_SampleState', ['sum_state', 'quantile_estimator_state'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['sum_params', 'quantile_estimator_params'])

  def __init__(self,
               initial_l2_norm_clip,
               noise_multiplier,
               target_unclipped_quantile,
               learning_rate,
               clipped_count_stddev,
               expected_num_records,
               geometric_update=True):
    """Initializes the QuantileAdaptiveClipSumQuery.

    Args:
      initial_l2_norm_clip: The initial value of clipping norm.
      noise_multiplier: The stddev of the noise added to the output will be this
        times the current value of the clipping norm.
      target_unclipped_quantile: The desired quantile of updates which should be
        unclipped. I.e., a value of 0.8 means a value of l2_norm_clip should be
        found for which approximately 20% of updates are clipped each round.
        Andrew et al. recommends that this be set to 0.5 to clip to the median.
      learning_rate: The learning rate for the clipping norm adaptation. With
        geometric updating, a rate of r means that the clipping norm will change
        by a maximum factor of exp(r) at each round. This maximum is attained
        when |actual_unclipped_fraction - target_unclipped_quantile| is 1.0.
        Andrew et al. recommends that this be set to 0.2 for geometric updating.
      clipped_count_stddev: The stddev of the noise added to the clipped_count.
        Andrew et al. recommends that this be set to `expected_num_records / 20`
        for reasonably fast adaptation and high privacy.
      expected_num_records: The expected number of records per round, used to
        estimate the clipped count quantile.
      geometric_update: If `True`, use geometric updating of clip (recommended).
    """
    self._noise_multiplier = noise_multiplier

    self._quantile_estimator_query = quantile_estimator_query.QuantileEstimatorQuery(
        initial_l2_norm_clip, target_unclipped_quantile, learning_rate,
        clipped_count_stddev, expected_num_records, geometric_update)

    self._sum_query = gaussian_query.GaussianSumQuery(
        initial_l2_norm_clip, noise_multiplier * initial_l2_norm_clip)

    assert isinstance(self._sum_query, dp_query.SumAggregationDPQuery)
    assert isinstance(self._quantile_estimator_query,
                      dp_query.SumAggregationDPQuery)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._GlobalState(
        tf.cast(self._noise_multiplier, tf.float32),
        self._sum_query.initial_global_state(),
        self._quantile_estimator_query.initial_global_state())

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._SampleParams(
        self._sum_query.derive_sample_params(global_state.sum_state),
        self._quantile_estimator_query.derive_sample_params(
            global_state.quantile_estimator_state))

  def initial_sample_state(self, template):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._SampleState(
        self._sum_query.initial_sample_state(template),
        self._quantile_estimator_query.initial_sample_state())

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    clipped_record, global_norm = (
        self._sum_query.preprocess_record_impl(params.sum_params, record))

    was_unclipped = self._quantile_estimator_query.preprocess_record(
        params.quantile_estimator_params, global_norm)

    return self._SampleState(clipped_record, was_unclipped)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_vectors, sum_state, sum_event = self._sum_query.get_noised_result(
        sample_state.sum_state, global_state.sum_state)
    del sum_state  # To be set explicitly later when we know the new clip.

    new_l2_norm_clip, new_quantile_estimator_state, quantile_event = (
        self._quantile_estimator_query.get_noised_result(
            sample_state.quantile_estimator_state,
            global_state.quantile_estimator_state))

    new_l2_norm_clip = tf.maximum(new_l2_norm_clip, 0.0)
    new_sum_stddev = new_l2_norm_clip * global_state.noise_multiplier
    new_sum_query_state = self._sum_query.make_global_state(
        l2_norm_clip=new_l2_norm_clip, stddev=new_sum_stddev)

    new_global_state = self._GlobalState(global_state.noise_multiplier,
                                         new_sum_query_state,
                                         new_quantile_estimator_state)

    event = dp_accounting.ComposedDpEvent(events=[sum_event, quantile_event])
    return noised_vectors, new_global_state, event

  def derive_metrics(self, global_state):
    """Returns the current clipping norm as a metric."""
    return collections.OrderedDict(clip=global_state.sum_state.l2_norm_clip)

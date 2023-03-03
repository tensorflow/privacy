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
"""`DPQuery` for tree aggregation queries with adaptive clipping."""

import collections

import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import quantile_estimator_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query


class QAdaClipTreeResSumQuery(dp_query.SumAggregationDPQuery):
  """`DPQuery` for tree aggregation queries with adaptive clipping.

  The implementation is based on tree aggregation noise for cumulative sum in
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  (https://arxiv.org/abs/2103.00039) and quantile-based adaptive clipping in
  "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871).

  The quantile value will be continuously estimated, but the clip norm is only
  updated when `reset_state` is called, and the tree state will be reset. This
  will force the clip norm (and corresponding stddev) in a tree unchanged.
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
               record_specs,
               target_unclipped_quantile,
               learning_rate,
               clipped_count_stddev,
               expected_num_records,
               geometric_update=True,
               noise_seed=None):
    """Initializes the `QAdaClipTreeResSumQuery`.

    Args:
      initial_l2_norm_clip: The initial value of clipping norm.
      noise_multiplier: The stddev of the noise added to the output will be this
        times the current value of the clipping norm.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
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
      noise_seed: Integer seed for the Gaussian noise generator of
        `TreeResidualSumQuery`. If `None`, a nondeterministic seed based on
        system time will be generated.
    """
    self._noise_multiplier = noise_multiplier

    self._quantile_estimator_query = (
        quantile_estimator_query.TreeQuantileEstimatorQuery(
            initial_l2_norm_clip,
            target_unclipped_quantile,
            learning_rate,
            clipped_count_stddev,
            expected_num_records,
            geometric_update,
        )
    )

    self._sum_query = (
        tree_aggregation_query.TreeResidualSumQuery.build_l2_gaussian_query(
            initial_l2_norm_clip,
            noise_multiplier,
            record_specs,
            noise_seed=noise_seed,
            use_efficient=True,
        )
    )

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
        self._sum_query.preprocess_record_l2_impl(params.sum_params, record))

    below_estimate = self._quantile_estimator_query.preprocess_record(
        params.quantile_estimator_params, global_norm)

    return self._SampleState(clipped_record, below_estimate)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_vectors, sum_state, sum_event = self._sum_query.get_noised_result(
        sample_state.sum_state, global_state.sum_state)

    _, quantile_estimator_state, quantile_event = (
        self._quantile_estimator_query.get_noised_result(
            sample_state.quantile_estimator_state,
            global_state.quantile_estimator_state))

    new_global_state = self._GlobalState(global_state.noise_multiplier,
                                         sum_state, quantile_estimator_state)
    event = dp_accounting.ComposedDpEvent(events=[sum_event, quantile_event])
    return noised_vectors, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree and updating the clip norm.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met. The clip norm (
    and corresponding noise stddev) for the tree aggregated sum query is only
    updated from the quantile-based estimation when `reset_state` is called.

    Args:
      noised_results: Noised cumulative sum returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        records noise for the conceptual cumulative sum of the current leaf
        node, and tree state for the next conceptual cumulative sum.

    Returns:
      New global state with restarted tree state, and new clip norm.
    """
    new_l2_norm_clip = tf.math.maximum(
        global_state.quantile_estimator_state.current_estimate, 0.0)
    new_sum_stddev = new_l2_norm_clip * global_state.noise_multiplier
    sum_state = self._sum_query.reset_l2_clip_gaussian_noise(
        global_state.sum_state,
        clip_norm=new_l2_norm_clip,
        stddev=new_sum_stddev)
    sum_state = self._sum_query.reset_state(noised_results, sum_state)
    quantile_estimator_state = self._quantile_estimator_query.reset_state(
        noised_results, global_state.quantile_estimator_state)

    return global_state._replace(
        sum_state=sum_state, quantile_estimator_state=quantile_estimator_state)

  def derive_metrics(self, global_state):
    """Returns the clipping norm and estimated quantile value as a metric."""
    return collections.OrderedDict(
        current_clip=global_state.sum_state.clip_value,
        estimate_clip=global_state.quantile_estimator_state.current_estimate)

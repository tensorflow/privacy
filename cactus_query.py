from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import distutils

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import dp_event
from tensorflow_privacy.privacy.dp_query import dp_query

import cactus_sampling


class CactusSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Clips records to bound the L2 norm, then adds Gaussian noise to the sum.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_clip', 'stddev'])

  def __init__(self, l2_norm_clip, stddev):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
    """
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev

  def make_global_state(self, l2_norm_clip, stddev):
    """Creates a global state from the given parameters."""
    return self._GlobalState(
        tf.cast(l2_norm_clip, tf.float32), tf.cast(stddev, tf.float32))

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self.make_global_state(self._l2_norm_clip, self._stddev)

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.l2_norm_clip

  def preprocess_record_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(record, clipped_as_list), norm

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    preprocessed_record, _ = self.preprocess_record_impl(params, record)
    return preprocessed_record

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    if distutils.version.LooseVersion(
        tf.__version__) < distutils.version.LooseVersion('2.0.0'):
      def add_noise(v):
        cactus_sample(stddev=stddev, size=tf.shape(input=v))
        return v + cactus_sample(stddev=stddev, size=tf.shape(input=v))
    else:
      def add_noise(v):
        return v + tf.cast(cactus_sample(stddev=stddev, size=tf.shape(input=v)),dtype=v.dtype)

    result = tf.nest.map_structure(add_noise, sample_state)
    noise_multiplier = global_state.stddev / global_state.l2_norm_clip
    event = dp_event.CactusDpEvent(noise_multiplier)

    return result, global_state, event

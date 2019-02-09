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
"""Differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.optimizers import gaussian_query


def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  if (tf.train.Optimizer.compute_gradients.__code__ is
      not cls.compute_gradients.__code__):
    tf.logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    def __init__(
        self,
        dp_average_query,
        num_microbatches,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._dp_average_query = dp_average_query
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_average_query.initial_global_state()
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):

      # Note: it would be closer to the correct i.i.d. sampling of records if
      # we sampled each microbatch from the appropriate binomial distribution,
      # although that still wouldn't be quite correct because it would be
      # sampling from the dataset without replacement.
      microbatches_losses = tf.reshape(loss, [self._num_microbatches, -1])
      sample_params = (
          self._dp_average_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        grads, _ = zip(*super(cls, self).compute_gradients(
            tf.gather(microbatches_losses, [i]), var_list, gate_gradients,
            aggregation_method, colocate_gradients_with_ops, grad_loss))
        grads_list = list(grads)
        sample_state = self._dp_average_query.accumulate_record(
            sample_params, sample_state, grads_list)
        return sample_state

      if var_list is None:
        var_list = (
            tf.trainable_variables() + tf.get_collection(
                tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
      sample_state = self._dp_average_query.initial_sample_state(
          self._global_state, var_list)

      if self._unroll_microbatches:
        for idx in range(self._num_microbatches):
          sample_state = process_microbatch(idx, sample_state)
      else:
        # Use of while_loop here requires that sample_state be a nested
        # structure of tensors. In general, we would prefer to allow it to be
        # an arbitrary opaque type.
        cond_fn = lambda i, _: tf.less(i, self._num_microbatches)
        body_fn = lambda i, state: [tf.add(i, 1), process_microbatch(i, state)]
        idx = tf.constant(0)
        _, sample_state = tf.while_loop(cond_fn, body_fn, [idx, sample_state])

      final_grads, self._global_state = (
          self._dp_average_query.get_noised_result(
              sample_state, self._global_state))

      return list(zip(final_grads, var_list))

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      dp_average_query = gaussian_query.GaussianAverageQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier, num_microbatches)
      if 'population_size' in kwargs:
        population_size = kwargs.pop('population_size')
        max_queries = kwargs.pop('ledger_max_queries', 1e6)
        max_samples = kwargs.pop('ledger_max_samples', 1e6)
        selection_probability = num_microbatches / population_size
        ledger = privacy_ledger.PrivacyLedger(
            population_size,
            selection_probability,
            max_samples,
            max_queries)
        dp_average_query = privacy_ledger.QueryWithLedger(
            dp_average_query, ledger)

      super(DPGaussianOptimizerClass, self).__init__(
          dp_average_query,
          num_microbatches,
          unroll_microbatches,
          *args,
          **kwargs)

  return DPGaussianOptimizerClass


DPAdagradOptimizer = make_optimizer_class(tf.train.AdagradOptimizer)
DPAdamOptimizer = make_optimizer_class(tf.train.AdamOptimizer)
DPGradientDescentOptimizer = make_optimizer_class(
    tf.train.GradientDescentOptimizer)

DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(
    tf.train.AdagradOptimizer)
DPAdamGaussianOptimizer = make_gaussian_optimizer_class(tf.train.AdamOptimizer)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(
    tf.train.GradientDescentOptimizer)


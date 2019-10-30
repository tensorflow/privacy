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

from absl import logging

import tensorflow as tf

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query


def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
  child_code = cls.compute_gradients.__code__
  GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        dp_sum_query: DPQuery object, specifying differential privacy
          mechanism to use.
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a tf.while_loop. Can be used if using a tf.while_loop
          raises an exception.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._dp_sum_query = dp_sum_query
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_sum_query.initial_global_state()
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      if callable(loss):
        # TF is running in Eager mode, check we received a vanilla tape.
        if not gradient_tape:
          raise ValueError('When in Eager mode, a tape needs to be passed.')

        vector_loss = loss()
        if self._num_microbatches is None:
          self._num_microbatches = tf.shape(input=vector_loss)[0]
        sample_state = self._dp_sum_query.initial_sample_state(var_list)
        microbatches_losses = tf.reshape(vector_loss,
                                         [self._num_microbatches, -1])
        sample_params = (
            self._dp_sum_query.derive_sample_params(self._global_state))

        def process_microbatch(i, sample_state):
          """Process one microbatch (record) with privacy helper."""
          microbatch_loss = tf.reduce_mean(
              input_tensor=tf.gather(microbatches_losses, [i]))
          grads = gradient_tape.gradient(microbatch_loss, var_list)
          sample_state = self._dp_sum_query.accumulate_record(
              sample_params, sample_state, grads)
          return sample_state

        for idx in range(self._num_microbatches):
          sample_state = process_microbatch(idx, sample_state)

        grad_sums, self._global_state = (
            self._dp_sum_query.get_noised_result(
                sample_state, self._global_state))

        def normalize(v):
          return v / tf.cast(self._num_microbatches, tf.float32)

        final_grads = tf.nest.map_structure(normalize, grad_sums)

        grads_and_vars = list(zip(final_grads, var_list))
        return grads_and_vars

      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')

        # Note: it would be closer to the correct i.i.d. sampling of records if
        # we sampled each microbatch from the appropriate binomial distribution,
        # although that still wouldn't be quite correct because it would be
        # sampling from the dataset without replacement.
        if self._num_microbatches is None:
          self._num_microbatches = tf.shape(input=loss)[0]

        microbatches_losses = tf.reshape(loss, [self._num_microbatches, -1])
        sample_params = (
            self._dp_sum_query.derive_sample_params(self._global_state))

        def process_microbatch(i, sample_state):
          """Process one microbatch (record) with privacy helper."""
          grads, _ = zip(*super(cls, self).compute_gradients(
              tf.reduce_mean(input_tensor=tf.gather(
                  microbatches_losses, [i])), var_list, gate_gradients,
              aggregation_method, colocate_gradients_with_ops, grad_loss))
          grads_list = [
              g if g is not None else tf.zeros_like(v)
              for (g, v) in zip(list(grads), var_list)
          ]
          sample_state = self._dp_sum_query.accumulate_record(
              sample_params, sample_state, grads_list)
          return sample_state

        if var_list is None:
          var_list = (
              tf.compat.v1.trainable_variables() + tf.compat.v1.get_collection(
                  tf.compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        sample_state = self._dp_sum_query.initial_sample_state(var_list)

        if self._unroll_microbatches:
          for idx in range(self._num_microbatches):
            sample_state = process_microbatch(idx, sample_state)
        else:
          # Use of while_loop here requires that sample_state be a nested
          # structure of tensors. In general, we would prefer to allow it to be
          # an arbitrary opaque type.
          cond_fn = lambda i, _: tf.less(i, self._num_microbatches)
          body_fn = lambda i, state: [tf.add(i, 1), process_microbatch(i, state)]  # pylint: disable=line-too-long
          idx = tf.constant(0)
          _, sample_state = tf.while_loop(
              cond=cond_fn, body=body_fn, loop_vars=[idx, sample_state])

        grad_sums, self._global_state = (
            self._dp_sum_query.get_noised_result(
                sample_state, self._global_state))

        def normalize(v):
          return tf.truediv(v, tf.cast(self._num_microbatches, tf.float32))

        final_grads = tf.nest.map_structure(normalize, grad_sums)

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
        num_microbatches=None,
        ledger=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)

      if ledger:
        dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                      ledger=ledger)

      super(DPGaussianOptimizerClass, self).__init__(
          dp_sum_query,
          num_microbatches,
          unroll_microbatches,
          *args,
          **kwargs)

    @property
    def ledger(self):
      return self._dp_sum_query.ledger

  return DPGaussianOptimizerClass

AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
AdamOptimizer = tf.compat.v1.train.AdamOptimizer
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

DPAdagradOptimizer = make_optimizer_class(AdagradOptimizer)
DPAdamOptimizer = make_optimizer_class(AdamOptimizer)
DPGradientDescentOptimizer = make_optimizer_class(GradientDescentOptimizer)

DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(AdagradOptimizer)
DPAdamGaussianOptimizer = make_gaussian_optimizer_class(AdamOptimizer)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(
    GradientDescentOptimizer)

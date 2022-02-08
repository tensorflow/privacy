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
"""Differentially private optimizers for TensorFlow."""

from absl import logging
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import gaussian_query


def make_optimizer_class(cls):
  """Given a subclass of `tf.compat.v1.train.Optimizer`, returns a DP-SGD subclass of it.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.compat.v1.train.Optimizer`.

  Returns:
    A DP-SGD subclass of `cls`.
  """
  parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__

  has_compute_gradients = hasattr(cls, 'compute_gradients')
  if has_compute_gradients:
    child_code = cls.compute_gradients.__code__
  GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
  if has_compute_gradients and child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = ("""Differentially private subclass of `{base_class}`.

       You can use this as a differentially private replacement for
       `{base_class}`. Note that you must ensure
       that any loss processed by this optimizer comes in vector
       form.

       This is the fully general form of the optimizer that allows you
       to define your own privacy mechanism. If you are planning to use
       the standard Gaussian mechanism, it is simpler to use the more
       specific `{gaussian_class}` class instead.

       When instantiating this optimizer, you need to supply several
       DP-related arguments followed by the standard arguments for
       `{short_base_class}`.

       Examples:

       ```python
       # Create GaussianSumQuery.
       dp_sum_query = gaussian_query.GaussianSumQuery(l2_norm_clip=1.0, stddev=0.5)

       # Create optimizer.
       opt = {dp_class}(dp_sum_query, 1, False, <standard arguments>)
       ```

       When using the optimizer, be sure to pass in the loss as a
       rank-one tensor with one entry for each example.

       ```python
       # Compute loss as a tensor. Do not call tf.reduce_mean as you
       # would with a standard optimizer.
       loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels=labels, logits=logits)

       train_op = opt.minimize(loss, global_step=global_step)
       ```

       """).format(
           base_class='tf.compat.v1.train.' + cls.__name__,
           gaussian_class='DP' +
           cls.__name__.replace('Optimizer', 'GaussianOptimizer'),
           short_base_class=cls.__name__,
           dp_class='DP' + cls.__name__)

    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        while_loop_parallel_iterations=10,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initializes the DPOptimizerClass.

      Args:
        dp_sum_query: `DPQuery` object, specifying differential privacy
          mechanism to use.
        num_microbatches: Number of microbatches into which each minibatch is
          split. If `None`, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a `tf.while_loop`. Can be used if using a
          `tf.while_loop` raises an exception.
        while_loop_parallel_iterations: The number of iterations allowed to run
          in parallel. It must be a positive integer. Applicable only when
          unroll_microbatches is set to False. It gives users some control over
          memory consumption.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self._dp_sum_query = dp_sum_query
      self._num_microbatches = num_microbatches
      self._global_state = None
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches
      self._while_loop_parallel_iterations = while_loop_parallel_iterations
      self._was_compute_gradients_called = False

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      """DP-SGD version of base class method."""
      self._was_compute_gradients_called = True
      if self._global_state is None:
        self._global_state = self._dp_sum_query.initial_global_state()

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
          with gradient_tape.stop_recording():
            grads = gradient_tape.gradient(microbatch_loss, var_list)
          sample_state = self._dp_sum_query.accumulate_record(
              sample_params, sample_state, grads)
          return sample_state

        for idx in range(self._num_microbatches):
          sample_state = process_microbatch(idx, sample_state)

        grad_sums, self._global_state, _ = (
            self._dp_sum_query.get_noised_result(sample_state,
                                                 self._global_state))

        def normalize(v):
          return v / tf.cast(self._num_microbatches, tf.float32)

        final_grads = tf.nest.map_structure(normalize, grad_sums)

        grads_and_vars = list(zip(final_grads, var_list))
        return grads_and_vars

      else:
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
          self_super = super(DPOptimizerClass, self)

          mean_loss = tf.reduce_mean(
              input_tensor=tf.gather(microbatches_losses, [i]))

          if hasattr(self_super, 'compute_gradients'):
            # This case covers optimizers in tf.train.
            compute_gradients_fn = self_super.compute_gradients
          else:
            # This case covers Keras optimizers from optimizers_v2.
            compute_gradients_fn = self_super._compute_gradients  # pylint: disable=protected-access

          if gradient_tape:
            # This is intended to work for TF2 and may not work for TF1.
            with gradient_tape.stop_recording():
              grads_list = list(gradient_tape.gradient(mean_loss, var_list))
          else:
            grads, _ = zip(*compute_gradients_fn(
                mean_loss, var_list, gate_gradients, aggregation_method,
                colocate_gradients_with_ops, grad_loss))
            grads_list = list(grads)

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
              cond=cond_fn,
              body=body_fn,
              loop_vars=[idx, sample_state],
              parallel_iterations=self._while_loop_parallel_iterations)

        grad_sums, self._global_state, _ = (
            self._dp_sum_query.get_noised_result(sample_state,
                                                 self._global_state))

        def normalize(v):
          try:
            return tf.truediv(v, tf.cast(self._num_microbatches, tf.float32))
          except TypeError:
            return None

        final_grads = tf.nest.map_structure(normalize, grad_sums)

        return list(zip(final_grads, var_list))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      # pylint: disable=g-doc-args, g-doc-return-or-yield
      """DP-SGD version of base class method."""
      assert self._was_compute_gradients_called, (
          'compute_gradients() on the differentially private optimizer was not'
          ' called. Which means that the training is not differentially '
          'private. It happens for example in Keras training in TensorFlow '
          '2.0+.')
      return super(DPOptimizerClass, self).apply_gradients(
          grads_and_vars=grads_and_vars, global_step=global_step, name=name)

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Given a subclass of `tf.compat.v1.train.Optimizer`, returns a subclass using DP-SGD with Gaussian averaging.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.compat.v1.train.Optimizer`.

  Returns:
    A subclass of `cls` using DP-SGD with Gaussian averaging.
  """

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):  # pylint: disable=empty-docstring
    __doc__ = ("""DP subclass of `{}`.

       You can use this as a differentially private replacement for
       `tf.compat.v1.train.{}`. This optimizer implements DP-SGD using
       the standard Gaussian mechanism.

       When instantiating this optimizer, you need to supply several
       DP-related arguments followed by the standard arguments for
       `{}`.

       Examples:

       ```python
       # Create optimizer.
       opt = {}(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1,
                <standard arguments>)
       ```

       When using the optimizer, be sure to pass in the loss as a
       rank-one tensor with one entry for each example.

       ```python
       # Compute loss as a tensor. Do not call tf.reduce_mean as you
       # would with a standard optimizer.
       loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels=labels, logits=logits)

       train_op = opt.minimize(loss, global_step=global_step)
       ```

       """).format(
           'tf.compat.v1.train.' + cls.__name__, cls.__name__, cls.__name__,
           'DP' + cls.__name__.replace('Optimizer', 'GaussianOptimizer'))

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      """Initializes the `DPGaussianOptimizerClass`.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch is
          split. If `None`, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a `tf.while_loop`. Can be used if using a
          `tf.while_loop` raises an exception.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._base_optimizer_class = cls

      dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)

      super(DPGaussianOptimizerClass,
            self).__init__(dp_sum_query, num_microbatches, unroll_microbatches,
                           *args, **kwargs)

    def get_config(self):
      """Creates configuration for Keras serialization.

      This method will be called when Keras creates model checkpoints
      and is necessary so that deserialization can be performed.

      Returns:
        A dict object storing arguments to be passed to the __init__ method
        upon deserialization.
      """

      config = self._base_optimizer_class.get_config(self)
      config.update({
          'l2_norm_clip': self._l2_norm_clip,
          'noise_multiplier': self._noise_multiplier,
          'num_microbatches': self._num_microbatches
      })

      return config

  return DPGaussianOptimizerClass


AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
AdamOptimizer = tf.compat.v1.train.AdamOptimizer
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
RMSPropOptimizer = tf.compat.v1.train.RMSPropOptimizer

DPAdagradOptimizer = make_optimizer_class(AdagradOptimizer)
DPAdamOptimizer = make_optimizer_class(AdamOptimizer)
DPGradientDescentOptimizer = make_optimizer_class(GradientDescentOptimizer)
DPRMSPropOptimizer = make_optimizer_class(RMSPropOptimizer)

DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(AdagradOptimizer)
DPAdamGaussianOptimizer = make_gaussian_optimizer_class(AdamOptimizer)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(
    GradientDescentOptimizer)
DPRMSPropGaussianOptimizer = make_gaussian_optimizer_class(RMSPropOptimizer)

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Differentially private version of Keras optimizer v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_privacy.privacy.dp_query import gaussian_query


def make_keras_optimizer_class(cls):
  """Given a subclass of `tf.keras.optimizers.Optimizer`, returns a DP-SGD subclass of it.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A DP-SGD subclass of `cls`.
  """

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = """Differentially private subclass of class `{base_class}`.

    You can use this as a differentially private replacement for
    `{base_class}`. This optimizer implements DP-SGD using
    the standard Gaussian mechanism.

    When instantiating this optimizer, you need to supply several
    DP-related arguments followed by the standard arguments for
    `{short_base_class}`.

    Examples:

    ```python
    # Create optimizer.
    opt = {dp_keras_class}(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1, 
            <standard arguments>)
    ```

    When using the optimizer, be sure to pass in the loss as a
    rank-one tensor with one entry for each example.

    The optimizer can be used directly via its `minimize` method, or
    through a Keras `Model`.

    ```python
    # Compute loss as a tensor by using tf.losses.Reduction.NONE.
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    # Use optimizer in a Keras model.
    opt.minimize(loss, var_list=[var])
    ```

    ```python
    # Compute loss as a tensor by using tf.losses.Reduction.NONE.
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    # Use optimizer in a Keras model.
    model = tf.keras.Sequential(...)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    model.fit(...)
    ```

    """.format(base_class='tf.keras.optimizers.' + cls.__name__,
               short_base_class=cls.__name__,
               dp_keras_class='DPKeras' + cls.__name__)

    # The class tf.keras.optimizers.Optimizer has two methods to compute
    # gradients, `_compute_gradients` and `get_gradients`. The first works
    # with eager execution, while the second runs in graph mode and is used
    # by canned estimators.

    # Internally, DPOptimizerClass stores hyperparameters both individually
    # and encapsulated in a `GaussianSumQuery` object for these two use cases.
    # However, this should be invisible to users of this class.

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch
          is split.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      self._global_state = None
      self._was_dp_gradients_called = False

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
      """DP-SGD version of base class method."""

      self._was_dp_gradients_called = True
      # Compute loss.
      if not callable(loss) and tape is None:
        raise ValueError('`tape` is required when a `Tensor` loss is passed.')
      tape = tape if tape is not None else tf.GradientTape()

      if callable(loss):
        with tape:
          if not callable(var_list):
            tape.watch(var_list)

          loss = loss()
          if self._num_microbatches is None:
            num_microbatches = tf.shape(input=loss)[0]
          else:
            num_microbatches = self._num_microbatches
          microbatch_losses = tf.reduce_mean(
              tf.reshape(loss, [num_microbatches, -1]), axis=1)

          if callable(var_list):
            var_list = var_list()
      else:
        with tape:
          if self._num_microbatches is None:
            num_microbatches = tf.shape(input=loss)[0]
          else:
            num_microbatches = self._num_microbatches
          microbatch_losses = tf.reduce_mean(
              tf.reshape(loss, [num_microbatches, -1]), axis=1)

      var_list = tf.nest.flatten(var_list)

      # Compute the per-microbatch losses using helpful jacobian method.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        jacobian = tape.jacobian(microbatch_losses, var_list)

        # Clip gradients to given l2_norm_clip.
        def clip_gradients(g):
          return tf.clip_by_global_norm(g, self._l2_norm_clip)[0]

        clipped_gradients = tf.map_fn(clip_gradients, jacobian)

        def reduce_noise_normalize_batch(g):
          # Sum gradients over all microbatches.
          summed_gradient = tf.reduce_sum(g, axis=0)

          # Add noise to summed gradients.
          noise_stddev = self._l2_norm_clip * self._noise_multiplier
          noise = tf.random.normal(
              tf.shape(input=summed_gradient), stddev=noise_stddev)
          noised_gradient = tf.add(summed_gradient, noise)

          # Normalize by number of microbatches and return.
          return tf.truediv(noised_gradient,
                            tf.cast(num_microbatches, tf.float32))

        final_gradients = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                clipped_gradients)

      return list(zip(final_gradients, var_list))

    def get_gradients(self, loss, params):
      """DP-SGD version of base class method."""

      self._was_dp_gradients_called = True
      if self._global_state is None:
        self._global_state = self._dp_sum_query.initial_global_state()

      # This code mostly follows the logic in the original DPOptimizerClass
      # in dp_optimizer.py, except that this returns only the gradients,
      # not the gradients and variables.
      microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])
      sample_params = (
          self._dp_sum_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        mean_loss = tf.reduce_mean(
            input_tensor=tf.gather(microbatch_losses, [i]))
        grads = tf.gradients(mean_loss, params)
        sample_state = self._dp_sum_query.accumulate_record(
            sample_params, sample_state, grads)
        return sample_state

      sample_state = self._dp_sum_query.initial_sample_state(params)
      for idx in range(self._num_microbatches):
        sample_state = process_microbatch(idx, sample_state)
      grad_sums, self._global_state = (
          self._dp_sum_query.get_noised_result(sample_state,
                                               self._global_state))

      def normalize(v):
        try:
          return tf.truediv(v, tf.cast(self._num_microbatches, tf.float32))
        except TypeError:
          return None

      final_grads = tf.nest.map_structure(normalize, grad_sums)

      return final_grads

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      """DP-SGD version of base class method."""
      assert self._was_dp_gradients_called, (
          'Neither _compute_gradients() or get_gradients() on the '
          'differentially private optimizer was called. This means the '
          'training is not differentially private. It may be the case that '
          'you need to upgrade to TF 2.4 or higher to use this particular '
          'optimizer.')
      return super(DPOptimizerClass,
                   self).apply_gradients(grads_and_vars, global_step, name)

  return DPOptimizerClass


DPKerasAdagradOptimizer = make_keras_optimizer_class(
    tf.keras.optimizers.Adagrad)
DPKerasAdamOptimizer = make_keras_optimizer_class(tf.keras.optimizers.Adam)
DPKerasSGDOptimizer = make_keras_optimizer_class(tf.keras.optimizers.SGD)

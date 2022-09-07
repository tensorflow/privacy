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

import tensorflow as tf

from tensorflow_privacy.privacy.dp_query import gaussian_query


def clip_gradients_vmap(g, l2_norm_clip):
  """Clips gradients in a way that is compatible with vectorized_map."""
  grads_flat = tf.nest.flatten(g)
  squared_l2_norms = [
      tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
  ]
  global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
  div = tf.maximum(global_norm / l2_norm_clip, 1.)
  clipped_flat = [g / div for g in grads_flat]
  clipped_grads = tf.nest.pack_sequence_as(g, clipped_flat)
  return clipped_grads


def make_vectorized_keras_optimizer_class(cls):
  """Given a subclass of `tf.keras.optimizers.Optimizer`, returns a vectorized DP-SGD subclass of it.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A vectorized DP-SGD subclass of `cls`.
  """

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = """Vectorized differentially private subclass of given class
    `{base_class}`.

    You can use this as a differentially private replacement for
    `{base_class}`. This optimizer implements DP-SGD using
    the standard Gaussian mechanism. It differs from `{dp_keras_class}` in that
    it attempts to vectorize the gradient computation and clipping of
    microbatches.

    When instantiating this optimizer, you need to supply several
    DP-related arguments followed by the standard arguments for
    `{short_base_class}`.

    Examples:

    ```python
    # Create optimizer.
    opt = {dp_vectorized_keras_class}(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1,
            <standard arguments>)
    ```

    When using the optimizer, be sure to pass in the loss as a
    rank-one tensor with one entry for each example.

    The optimizer can be used directly via its `minimize` method, or
    through a Keras `Model`.

    ```python
    # Compute loss as a tensor by using tf.losses.Reduction.NONE.
    # Compute vector of per-example loss rather than its mean over a minibatch.
    # (Side note: Always verify that the output shape when using
    # tf.losses.Reduction.NONE-- it can sometimes be surprising.
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

    """.format(
        base_class='tf.keras.optimizers.' + cls.__name__,
        dp_keras_class='DPKeras' + cls.__name__,
        short_base_class=cls.__name__,
        dp_vectorized_keras_class='VectorizedDPKeras' + cls.__name__)

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        unconnected_gradients_to_zero=False,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch is
          split.
        unconnected_gradients_to_zero: The Jacobian is used to compute the
          microbatch losses. If a node in the graph is deliberately not
          connected, then the Jacobian computation will return a `None` for that
          node. Set this flag to True to treat these Jacobians as zero.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._unconnected_gradients_to_zero = unconnected_gradients_to_zero
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
        jacobian = tape.jacobian(
            microbatch_losses,
            var_list,
            unconnected_gradients=(tf.UnconnectedGradients.ZERO
                                   if self._unconnected_gradients_to_zero else
                                   tf.UnconnectedGradients.NONE))

        clipped_gradients = tf.vectorized_map(
            lambda g: clip_gradients_vmap(g, self._l2_norm_clip), jacobian)

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

      if self._num_microbatches is None:
        num_microbatches = tf.shape(input=loss)[0]
      else:
        num_microbatches = self._num_microbatches

      microbatch_losses = tf.reshape(loss, [num_microbatches, -1])

      def process_microbatch(microbatch_loss):
        """Compute clipped grads for one microbatch."""
        mean_loss = tf.reduce_mean(input_tensor=microbatch_loss)
        grads = cls.get_gradients(self, mean_loss, params)
        grads_list = [
            g if g is not None else tf.zeros_like(v)
            for (g, v) in zip(list(grads), params)
        ]
        clipped_grads = clip_gradients_vmap(grads_list, self._l2_norm_clip)
        return clipped_grads

      clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses)

      def reduce_noise_normalize_batch(stacked_grads):
        summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
        noise_stddev = self._l2_norm_clip * self._noise_multiplier
        noise = tf.random.normal(
            tf.shape(input=summed_grads), stddev=noise_stddev)
        noised_grads = summed_grads + noise
        return noised_grads / tf.cast(num_microbatches, tf.float32)

      final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
                                          clipped_grads)
      return final_grads

    def apply_gradients(self, *args, **kwargs):
      """DP-SGD version of base class method."""
      assert self._was_dp_gradients_called, (
          'Neither _compute_gradients() or get_gradients() on the '
          'differentially private optimizer was called. This means the '
          'training is not differentially private. It may be the case that '
          'you need to upgrade to TF 2.4 or higher to use this particular '
          'optimizer.')
      return super(DPOptimizerClass, self).apply_gradients(*args, **kwargs)

  return DPOptimizerClass


VectorizedDPKerasAdagradOptimizer = make_vectorized_keras_optimizer_class(
    tf.keras.optimizers.Adagrad)
VectorizedDPKerasAdamOptimizer = make_vectorized_keras_optimizer_class(
    tf.keras.optimizers.Adam)
VectorizedDPKerasSGDOptimizer = make_vectorized_keras_optimizer_class(
    tf.keras.optimizers.SGD)

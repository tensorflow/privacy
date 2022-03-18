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

    In DP-SGD training, a larger batch size typically helps to achieve better
    privacy/utility tradeoff. However there is typically a maximum batch size
    imposed by hardware.
    This optimizer can emulate large batch sizes on hardware with limited
    memory by accumulating gradients for several steps before actually
    applying them to update model weights.
    Constructor argument `gradient_accumulation_steps` controls the number
    of steps for which gradients are accumulated before updating
    the model weights.

    Below is an example which demonstrates how to use this feature:

    ```python
    # Create optimizer which will be accumulating gradients for 4 steps.
    # and then performing an update of model weights.
    opt = {dp_keras_class}(l2_norm_clip=1.0,
                           noise_multiplier=0.5,
                           num_microbatches=1,
                           gradient_accumulation_steps=4,
                           <standard arguments>)

    # Use optimizer in a regular way.
    # First three calls to opt.minimize won't update model weights and will
    # only accumulate gradients. Model weights will be updated on the fourth
    # call to opt.minimize
    opt.minimize(loss, var_list=[var])
    ```

    Note that when using this feature effective batch size is
    `gradient_accumulation_steps * one_step_batch_size` where
    `one_step_batch_size` size of the batch which is passed to single step
    of the optimizer. Thus user may have to adjust learning rate, weight decay
    and possibly other training hyperparameters accordingly.
    """.format(
        base_class='tf.keras.optimizers.' + cls.__name__,
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
        gradient_accumulation_steps=1,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch is
          split. Default is `None` which means that number of microbatches
          is equal to batch size (i.e. each microbatch contains exactly one
          example). If `gradient_accumulation_steps` is greater than 1 and
          `num_microbatches` is not `None` then the effective number of
          microbatches is equal to
          `num_microbatches * gradient_accumulation_steps`.
        gradient_accumulation_steps: If greater than 1 then optimizer will be
          accumulating gradients for this number of optimizer steps before
          applying them to update model weights. If this argument is set to 1
          then updates will be applied on each optimizer step.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self.gradient_accumulation_steps = gradient_accumulation_steps
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      self._global_state = None
      self._was_dp_gradients_called = False

    def _create_slots(self, var_list):
      super()._create_slots(var_list)  # pytype: disable=attribute-error
      if self.gradient_accumulation_steps > 1:
        for var in var_list:
          self.add_slot(var, 'grad_acc')

    def _prepare_local(self, var_device, var_dtype, apply_state):
      super()._prepare_local(var_device, var_dtype, apply_state)  # pytype: disable=attribute-error
      if self.gradient_accumulation_steps > 1:
        apply_update = tf.math.equal(
            tf.math.floormod(self.iterations + 1,
                             self.gradient_accumulation_steps), 0)
        grad_scaler = tf.cast(1. / self.gradient_accumulation_steps, var_dtype)
        apply_state[(var_device, var_dtype)].update({
            'apply_update': apply_update,
            'grad_scaler': grad_scaler
        })

    def _resource_apply_dense(self, grad, var, apply_state=None):
      if self.gradient_accumulation_steps > 1:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))
        grad_acc = self.get_slot(var, 'grad_acc')

        def _update_grad():
          apply_grad_op = super(DPOptimizerClass, self)._resource_apply_dense(
              grad_acc + grad * coefficients['grad_scaler'], var, apply_state)  # pytype: disable=attribute-error
          with tf.control_dependencies([apply_grad_op]):
            return grad_acc.assign(
                tf.zeros_like(grad_acc),
                use_locking=self._use_locking,
                read_value=False)

        def _accumulate():
          return grad_acc.assign_add(
              grad * coefficients['grad_scaler'],
              use_locking=self._use_locking,
              read_value=False)

        return tf.cond(coefficients['apply_update'], _update_grad, _accumulate)
      else:
        return super()._resource_apply_dense(grad, var, apply_state)  # pytype: disable=attribute-error

    def _resource_apply_sparse_duplicate_indices(self, *args, **kwargs):
      if self.gradient_accumulation_steps > 1:
        raise NotImplementedError(
            'Sparse gradients are not supported with large batch emulation.')
      else:
        return super()._resource_apply_sparse_duplicate_indices(*args, **kwargs)  # pytype: disable=attribute-error

    def _resource_apply_sparse(self, *args, **kwargs):
      if self.gradient_accumulation_steps > 1:
        raise NotImplementedError(
            'Sparse gradients are not supported with large batch emulation.')
      else:
        return super()._resource_apply_sparse(*args, **kwargs)  # pytype: disable=attribute-error

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
            microbatch_losses, var_list, unconnected_gradients='zero')

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
      grad_sums, self._global_state, _ = (
          self._dp_sum_query.get_noised_result(sample_state,
                                               self._global_state))

      def normalize(v):
        try:
          return tf.truediv(v, tf.cast(self._num_microbatches, tf.float32))
        except TypeError:
          return None

      final_grads = tf.nest.map_structure(normalize, grad_sums)

      return final_grads

    def get_config(self):
      """Returns the config of the optimizer.

      An optimizer config is a Python dictionary (serializable)
      containing the configuration of an optimizer.
      The same optimizer can be reinstantiated later
      (without any saved state) from this configuration.

      Returns:
          Python dictionary.
      """
      config = super().get_config()
      config.update({
          'l2_norm_clip': self._l2_norm_clip,
          'noise_multiplier': self._noise_multiplier,
          'num_microbatches': self._num_microbatches,
      })
      return config

    def apply_gradients(self, *args, **kwargs):
      """DP-SGD version of base class method."""
      assert self._was_dp_gradients_called, (
          'Neither _compute_gradients() or get_gradients() on the '
          'differentially private optimizer was called. This means the '
          'training is not differentially private. It may be the case that '
          'you need to upgrade to TF 2.4 or higher to use this particular '
          'optimizer.')
      return super().apply_gradients(*args, **kwargs)

  return DPOptimizerClass


DPKerasAdagradOptimizer = make_keras_optimizer_class(
    tf.keras.optimizers.Adagrad)
DPKerasAdamOptimizer = make_keras_optimizer_class(tf.keras.optimizers.Adam)
DPKerasSGDOptimizer = make_keras_optimizer_class(tf.keras.optimizers.SGD)

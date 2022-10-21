# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Differentially private version of Keras optimizer v2 supporting sparse gradient updates."""

import tensorflow as tf

from tensorflow_privacy.privacy.optimizers import clip_and_aggregate_gradients as cag


# This parameter is used by clip_and_aggregate_gradients to determine when to
# switch between sparse and dense representation. See the comments there
# for details. Here we expose this parameter internally to allow potential
# adjustment.
_KEEP_SPARSE_THRESHOLD = 10000


def make_sparse_keras_optimizer_class(cls):
  """Given a subclass of `tf.keras.optimizers.legacy.Optimizer`, returns a DP-SGD subclass of it supporting sparse gradient updates.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.keras.optimizers.legacy.Optimizer`.

  Returns:
    A DP-SGD subclass of `cls`, similar to those defined in
    dp_optimizer_keras, which uses microbatch and gradient accumulation
    to emulate large batch size for training high quality DP models. But
    compared to the optimizers there, there are several significant
    differences.
    1. The optimizers here only support the TF2 interface of `minimize`.
    2. The noise is only added when the gradients are applied, but not at
       each intermediate computation, i.e. in `_compute_gradients`. This
       has a few consequences: first, if one calls _compute_gradients, the
       returned gradients have no noise added; secondly, the noise added
       to each effective batch is noise_multiplier * l2_norm_clip,
       instead of
       sqrt(gradient_accumulation_steps) * noise_multiplier * l2_norm_clip.
    3. The optimizers support sparse gradient representation which is much
       more memory efficient. Hence, it can support larger value of
       `num_microbatches`. Together with the sparse updates, this provides
       significant speedup over the previous optimizers.
  """

  class DPOptimizerClass(cls):  # pylint: disable=missing-class-docstring
    __doc__ = """Differentially private subclass of class `{base_class}`.

    You can use this as a differentially private replacement for
    `{base_class}`. This optimizer implements DP-SGD using
    the standard Gaussian mechanism. Note: This optimizer provides
    more efficient updates for sparse models with large embedding variables
    where each training example only touches a small number of embeddings.
    It only supports TF2 and `minimize` method.

    When instantiating this optimizer, you need to supply several
    DP-related arguments followed by the standard arguments for
    `{short_base_class}`.

    Examples:

    ```python
    # Create optimizer.
    opt = {dp_keras_class}(l2_norm_clip=1.0, noise_multiplier=0.5,
           num_microbatches=1, <standard arguments>)
    ```

    When using the optimizer, be sure to pass in the loss as a
    rank-one tensor with one entry for each example.

    The optimizer can be used directly via its `minimize` method, or
    through a Keras `Model`.

    ```python
    # Computes loss as a tensor by using tf.losses.Reduction.NONE.
    # Computes vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    # Use optimizer in a Keras model.
    opt.minimize(loss, var_list=[var])
    ```

    ```python
    # Computes loss as a tensor by using tf.losses.Reduction.NONE.
    # Computes vector of per-example loss rather than its mean over a minibatch.
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

    Note that when using this feature, the effective batch size is
    `gradient_accumulation_steps * one_step_batch_size` where
    `one_step_batch_size` is the size of the batch passed to single step of
    the optimizer. Thus user may have to adjust learning rate, weight decay
    and possibly other training hyperparameters accordingly.

    Additionally, user may need to adjust the batch size in the data generator,
    or the number of calls to the data generator, depending on the training
    framework used. For example, when using Keras model.fit(...) with a
    user-defined data generator, one may need to make the data generator return
    `one_step_batch_size` examples each time, and scale the `steps_per_epoch`
    by `gradient_accumulation_steps`. This is because the data generator is
    called `steps_per_epoch` times per epoch, and one call only returns
    `one_step_batch_size` (instead of `effective_batch_size`) examples now.
    """.format(
        base_class='tf.keras.optimizers.legacy' + cls.__name__,
        short_base_class=cls.__name__,
        dp_keras_class='DPKeras' + cls.__name__)

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        gradient_accumulation_steps=1,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initializes the DPOptimizerClass.

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
      self._was_dp_gradients_called = False
      self._noise_stddev = None
      if self._num_microbatches is not None:
        # The loss/gradients is the mean over the microbatches so we
        # divide the noise by num_microbatches too to obtain the correct
        # normalized noise.  If _num_microbatches is not set, the noise stddev
        # will be set later when the loss is given.
        self._noise_stddev = (self._l2_norm_clip * self._noise_multiplier /
                              self._num_microbatches)

    def _generate_noise(self, g):
      """Returns noise to be added to `g`."""
      if self._noise_stddev is None:
        raise ValueError('noise_stddev is not set yet.')
      return tf.random.normal(tf.shape(input=g), stddev=self._noise_stddev)

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

    def _resource_apply(self, accum_op, grad, var, apply_state=None):
      """Help method for _resource_apply_dense and _resource_apply_sparse."""
      if self.gradient_accumulation_steps > 1:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))
        grad_acc = self.get_slot(var, 'grad_acc')

        def _update_grad():
          noisy_grad = grad_acc + self._generate_noise(grad_acc)
          apply_grad_op = super(DPOptimizerClass, self)._resource_apply_dense(
              noisy_grad * coefficients['grad_scaler'],
              var, apply_state)  # pytype: disable=attribute-error
          with tf.control_dependencies([apply_grad_op]):
            return grad_acc.assign(
                tf.zeros_like(grad_acc),
                use_locking=self._use_locking,
                read_value=False)
        accum_op(grad_acc, grad, use_locking=self._use_locking)
        return tf.cond(
            coefficients['apply_update'], _update_grad, lambda: tf.no_op())  # pylint: disable=unnecessary-lambda
      else:
        grad = tf.convert_to_tensor(grad)
        grad = grad + self._generate_noise(grad)
        return super()._resource_apply_dense(
            grad, var, apply_state)  # pytype: disable=attribute-error

    def _resource_apply_dense(self, grad, var, apply_state=None):
      """Handles dense gradients."""
      def _accum_op(grad_acc, grad, use_locking):
        return grad_acc.assign_add(
            grad, use_locking=use_locking, read_value=False)
      return self._resource_apply(_accum_op, grad, var, apply_state)

    # This method is implemented the same as that in optimizer_v2.py. We
    # redefine it here because it gets overridden by the SGD optimizer (and
    # potentially other optimizers too). If we omit it, it would cause an error
    # if the parent optimizer is the SGD optimizer.
    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, **kwargs):
      """Handles sparse gradients."""
      def _deduplicate_indexed_slices(values, indices):
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.math.unsorted_segment_sum(
            values, new_index_positions, tf.shape(unique_indices)[0]
        )
        return (summed_values, unique_indices)
      summed_grad, unique_indices = _deduplicate_indexed_slices(
          values=grad, indices=indices)
      return self._resource_apply_sparse(
          summed_grad, var, unique_indices, **kwargs)  # pytype: disable=attribute-error

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
      """Handles deduped sparse gradients."""
      def _accum_op(grad_acc, sparse_delta, use_locking):
        return grad_acc.scatter_add(
            sparse_delta=sparse_delta, use_locking=use_locking)
      sparse_delta = tf.IndexedSlices(
          values=grad, indices=indices, dense_shape=var.shape)
      return self._resource_apply(_accum_op, sparse_delta, var, apply_state)

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
      """DP-SGD version of base class method."""
      self._was_dp_gradients_called = True

      # Computes loss.
      if not callable(loss) and tape is None:
        raise ValueError('`tape` is required when a `Tensor` loss is passed.')

      tape = tape if tape is not None else tf.GradientTape()

      with tape:
        if callable(loss):
          if not callable(var_list):
            tape.watch(var_list)

          loss = loss()

        if self._num_microbatches is None:
          num_microbatches = tf.shape(input=loss)[0]
          self._noise_stddev = tf.divide(
              self._l2_norm_clip * self._noise_multiplier,
              tf.cast(num_microbatches, tf.float32))
        else:
          num_microbatches = self._num_microbatches
        microbatch_losses = tf.reduce_mean(
            tf.reshape(loss, [num_microbatches, -1]), axis=1)

        if callable(var_list):
          var_list = var_list()

      var_list = tf.nest.flatten(var_list)

      # Computes and aggregates per-microbatch clipped gradients.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        gradients = cag.clip_and_aggregate_gradients(
            tape, microbatch_losses, var_list,
            unconnected_gradients='zero',
            l2_norm_clip=self._l2_norm_clip,
            normalize=False,
            aggregate_method='mean',
            keep_sparse_threshold=_KEEP_SPARSE_THRESHOLD)
        return list(zip(gradients, var_list))

    def get_gradients(self, loss, params):
      """DP-SGD version of base class method."""
      raise ValueError('Only _compute_gradients is supported.')

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


DPSparseKerasAdagradOptimizer = make_sparse_keras_optimizer_class(
    tf.keras.optimizers.legacy.Adagrad)
DPSparseKerasAdamOptimizer = make_sparse_keras_optimizer_class(
    tf.keras.optimizers.legacy.Adam)
DPSparseKerasSGDOptimizer = make_sparse_keras_optimizer_class(
    tf.keras.optimizers.legacy.SGD)

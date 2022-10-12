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
from typing import List, Optional, Type, Union
import warnings

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import restart_query
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query

_VarListType = List[Union[tf.Tensor, tf.Variable]]


def _normalize(microbatch_gradient: tf.Tensor,
               num_microbatches: float) -> tf.Tensor:
  """Normalizes `microbatch_gradient` by `num_microbatches`."""
  return tf.truediv(microbatch_gradient,
                    tf.cast(num_microbatches, microbatch_gradient.dtype))


def make_keras_generic_optimizer_class(
    cls: Type[tf.keras.optimizers.Optimizer]):
  """Returns a differentially private (DP) subclass of `cls`.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.keras.optimizers.legacy.Optimizer`.

  Returns:
    A generic DP-SGD subclass of `cls`, compatible with many DP queries.
  """

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring,missing-class-docstring
    __doc__ = """Differentially private subclass of class `{base_class}`.

    You can use this as a differentially private replacement for
    `{base_class}`. This optimizer implements a differentiallyy private version
    of the stochastic gradient descent optimizer `cls` using the chosen
    `dp_query.DPQuery` instance.

    When instantiating this optimizer, you need to supply several
    DP-related arguments followed by the standard arguments for
    `{short_base_class}`.

    Examples:

    ```python
    # Create optimizer.
    gaussian_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1
    )
    opt = {dp_keras_class}(dp_sum_query=gaussian_query, <standard arguments>)
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
    gaussian_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1
    )
    opt = {dp_keras_class}(dp_sum_query=gaussian_query,
                           num_microbatches=1,
                           gradient_accumulation_steps=4,
                           <standard arguments>)

    # Use optimizer in a regular way.
    # First three calls to opt.minimize won't update model weights and will
    # only accumulate gradients. Model weights will be updated on the fourth
    # call to opt.minimize
    opt.minimize(loss, var_list=[var])
    ```

    Note that when using this feature,
    1. effective batch size is `gradient_accumulation_steps * one_step_batch_size`
      where `one_step_batch_size` is the size of the batch passed to single step
      of the optimizer. Thus user may have to adjust learning rate, weight decay
      and possibly other training hyperparameters accordingly.
    2. effective noise (the noise to be used for privacy computation) is
       `noise_multiplier * sqrt(gradient_accumulation_steps)`, as the optimizer
       adds noise of `self._noise_multiplier` to every step. Thus user may have
       to adjust the `noise_multiplier` or the privacy computation.
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

    # The class tf.keras.optimizers.legacy.Optimizer has two methods to compute
    # gradients, `_compute_gradients` and `get_gradients`. The first works
    # with eager execution, while the second runs in graph mode and is used
    # by canned estimators.

    # Internally, DPOptimizerClass stores hyperparameters both individually
    # and encapsulated in a `GaussianSumQuery` object for these two use cases.
    # However, this should be invisible to users of this class.

    def __init__(
        self,
        dp_sum_query: dp_query.DPQuery,
        num_microbatches: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initializes the DPOptimizerClass.

      Args:
        dp_sum_query: `DPQuery` object, specifying differential privacy
          mechanism to use.
        num_microbatches: Number of microbatches into which each minibatch is
          split. Default is `None` which means that number of microbatches is
          equal to batch size (i.e. each microbatch contains exactly one
          example). If `gradient_accumulation_steps` is greater than 1 and
          `num_microbatches` is not `None` then the effective number of
          microbatches is equal to `num_microbatches *
          gradient_accumulation_steps`.
        gradient_accumulation_steps: If greater than 1 then optimizer will be
          accumulating gradients for this number of optimizer steps before
          applying them to update model weights. If this argument is set to 1
          then updates will be applied on each optimizer step.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self.gradient_accumulation_steps = gradient_accumulation_steps
      self._num_microbatches = num_microbatches
      self._dp_sum_query = dp_sum_query
      self._was_dp_gradients_called = False
      # We initialize here for `_compute_gradients` because of requirements from
      # the tf.keras.Model API. Specifically, keras models use the
      # `_compute_gradients` method for both eager and graph mode. So,
      # instantiating the state here is necessary to avoid graph compilation
      # issues.

      self._global_state = self._dp_sum_query.initial_global_state()

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

      with tape:
        if callable(loss):
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

      var_list = tf.nest.flatten(var_list)

      sample_params = (
          self._dp_sum_query.derive_sample_params(self._global_state))

      # Compute the per-microbatch losses using helpful jacobian method.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        jacobian_per_var = tape.jacobian(
            microbatch_losses, var_list, unconnected_gradients='zero')

        def process_microbatch(sample_state, microbatch_jacobians):
          """Process one microbatch (record) with privacy helper."""
          sample_state = self._dp_sum_query.accumulate_record(
              sample_params, sample_state, microbatch_jacobians)
          return sample_state

        sample_state = self._dp_sum_query.initial_sample_state(var_list)

        def body_fn(idx, sample_state):
          microbatch_jacobians_per_var = [
              jacobian[idx] for jacobian in jacobian_per_var
          ]
          sample_state = process_microbatch(sample_state,
                                            microbatch_jacobians_per_var)
          return tf.add(idx, 1), sample_state

        cond_fn = lambda idx, _: tf.less(idx, num_microbatches)
        idx = tf.constant(0)
        _, sample_state = tf.while_loop(cond_fn, body_fn, [idx, sample_state])

        grad_sums, self._global_state, _ = (
            self._dp_sum_query.get_noised_result(sample_state,
                                                 self._global_state))
        final_grads = tf.nest.map_structure(_normalize, grad_sums,
                                            [num_microbatches] * len(grad_sums))

      return list(zip(final_grads, var_list))

    def get_gradients(self, loss, params):
      """DP-SGD version of base class method."""
      if not self._was_dp_gradients_called:
        # We create the global state here due to tf.Estimator API requirements,
        # specifically, that instantiating the global state outside this
        # function leads to graph compilation errors of attempting to capture an
        # EagerTensor.
        self._global_state = self._dp_sum_query.initial_global_state()
        self._was_dp_gradients_called = True

      # This code mostly follows the logic in the original DPOptimizerClass
      # in dp_optimizer.py, except that this returns only the gradients,
      # not the gradients and variables.
      if self._num_microbatches is None:
        num_microbatches = tf.shape(input=loss)[0]
      else:
        num_microbatches = self._num_microbatches

      microbatch_losses = tf.reshape(loss, [num_microbatches, -1])
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

      def body_fn(idx, sample_state):
        sample_state = process_microbatch(idx, sample_state)
        return tf.add(idx, 1), sample_state

      cond_fn = lambda idx, _: tf.less(idx, num_microbatches)
      idx = tf.constant(0)
      _, sample_state = tf.while_loop(cond_fn, body_fn, [idx, sample_state])
      grad_sums, self._global_state, _ = (
          self._dp_sum_query.get_noised_result(sample_state,
                                               self._global_state))

      final_grads = tf.nest.map_structure(_normalize, grad_sums,
                                          [num_microbatches] * len(grad_sums))

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
          'global_state': self._global_state._asdict(),
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


def make_gaussian_query_optimizer_class(cls):
  """Returns a differentially private optimizer using the `GaussianSumQuery`.

  Args:
    cls: `DPOptimizerClass`, the output of `make_keras_optimizer_class`.

  Returns:
    A DP-SGD subclass of `cls` using the `GaussianQuery`, the canonical DP-SGD
    implementation.
  """

  def return_gaussian_query_optimizer(
      l2_norm_clip: float,
      noise_multiplier: float,
      num_microbatches: Optional[int] = None,
      gradient_accumulation_steps: int = 1,
      *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
      **kwargs):
    """Returns a `DPOptimizerClass` `cls` using the `GaussianSumQuery`.

    This function is a thin wrapper around
    `make_keras_optimizer_class.<locals>.DPOptimizerClass` which can be used to
    apply a `GaussianSumQuery` to any `DPOptimizerClass`.

    When combined with stochastic gradient descent, this creates the canonical
    DP-SGD algorithm of "Deep Learning with Differential Privacy"
    (see https://arxiv.org/abs/1607.00133).

    When instantiating this optimizer, you need to supply several
    DP-related arguments followed by the standard arguments for
    `{short_base_class}`.

    As an example, see the below or the documentation of the DPOptimizerClass.

    ```python
    # Create optimizer.
    opt = {dp_keras_class}(l2_norm_clip=1.0, noise_multiplier=0.5,
        num_microbatches=1, <standard arguments>)
    ```

    Args:
      l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
      noise_multiplier: Ratio of the standard deviation to the clipping norm.
      num_microbatches: Number of microbatches into which each minibatch is
        split. Default is `None` which means that number of microbatches is
        equal to batch size (i.e. each microbatch contains exactly one example).
        If `gradient_accumulation_steps` is greater than 1 and
        `num_microbatches` is not `None` then the effective number of
        microbatches is equal to `num_microbatches *
        gradient_accumulation_steps`.
      gradient_accumulation_steps: If greater than 1 then optimizer will be
        accumulating gradients for this number of optimizer steps before
        applying them to update model weights. If this argument is set to 1 then
        updates will be applied on each optimizer step.
      *args: These will be passed on to the base class `__init__` method.
      **kwargs: These will be passed on to the base class `__init__` method.
    """
    dp_sum_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip, l2_norm_clip * noise_multiplier)
    return cls(
        dp_sum_query=dp_sum_query,
        num_microbatches=num_microbatches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        *args,
        **kwargs)

  return return_gaussian_query_optimizer


def make_dpftrl_tree_aggregation_optimizer_class(cls):
  """Returns a differentially private follow-the-regularized-leader optimizer.

  Args:
    cls: `DPOptimizerClass`, the output of `make_keras_optimizer_class`.
  """

  def return_dpftrl_tree_aggregation_optimizer(
      l2_norm_clip: float,
      noise_multiplier: float,
      var_list_or_model: Union[_VarListType, tf.keras.Model],
      num_microbatches: Optional[int] = None,
      gradient_accumulation_steps: int = 1,
      restart_period: Optional[int] = None,
      restart_warmup: Optional[int] = None,
      noise_seed: Optional[int] = None,
      *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
      **kwargs):
    """Returns a `DPOptimizerClass` `cls` using the `TreeAggregationQuery`.

    Combining this query with a SGD optimizer can be used to implement the
    DP-FTRL algorithm in
    "Practical and Private (Deep) Learning without Sampling or Shuffling".

    This function is a thin wrapper around
    `make_keras_optimizer_class.<locals>.DPOptimizerClass` which can be used to
    apply a `TreeAggregationQuery` to any `DPOptimizerClass`.

    Args:
      l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
      noise_multiplier: Ratio of the standard deviation to the clipping norm.
      var_list_or_model: Either a tf.keras.Model or a list of tf.variables from
        which `tf.TensorSpec`s can be defined. These specify the structure and
        shapes of records (gradients).
      num_microbatches: Number of microbatches into which each minibatch is
        split. Default is `None` which means that number of microbatches is
        equal to batch size (i.e. each microbatch contains exactly one example).
        If `gradient_accumulation_steps` is greater than 1 and
        `num_microbatches` is not `None` then the effective number of
        microbatches is equal to `num_microbatches *
        gradient_accumulation_steps`.
      gradient_accumulation_steps: If greater than 1 then optimizer will be
        accumulating gradients for this number of optimizer steps before
        applying them to update model weights. If this argument is set to 1 then
        updates will be applied on each optimizer step.
      restart_period: (Optional) Restart wil occur after `restart_period` steps.
        The default (None) means there will be no periodic restarts. Must be a
        positive integer. If `restart_warmup` is passed, this only applies to
        the second restart and onwards and must be not None.
      restart_warmup: (Optional) The first restart will occur after
        `restart_warmup` steps. The default (None) means no warmup. Must be an
        integer in the range [1, `restart_period` - 1].
      noise_seed: (Optional) Integer seed for the Gaussian noise generator. If
        `None`, a nondeterministic seed based on system time will be generated.
      *args: These will be passed on to the base class `__init__` method.
      **kwargs: These will be passed on to the base class `__init__` method.
    Raise:
      ValueError: If restart_warmup is not None and restart_period is None.
    """
    if restart_warmup is not None and restart_period is None:
      raise ValueError(
          '`restart_period` was None when `restart_warmup` was not None.')

    if isinstance(var_list_or_model, tf.keras.layers.Layer):
      model_trainable_specs = tf.nest.map_structure(
          lambda t: tf.TensorSpec(t.shape),
          var_list_or_model.trainable_variables)
    else:
      model_trainable_specs = tf.nest.map_structure(
          lambda t: tf.TensorSpec(tf.shape(t)), var_list_or_model)

    if restart_period is not None:
      sum_query = (
          tree_aggregation_query.TreeResidualSumQuery.build_l2_gaussian_query(
              l2_norm_clip, noise_multiplier, model_trainable_specs,
              noise_seed))
      restart_indicator = restart_query.PeriodicRoundRestartIndicator(
          period=restart_period, warmup=restart_warmup)
      tree_aggregation_sum_query = restart_query.RestartQuery(
          sum_query, restart_indicator)
    else:
      tree_aggregation_sum_query = (
          tree_aggregation_query.TreeResidualSumQuery.build_l2_gaussian_query(
              l2_norm_clip, noise_multiplier, model_trainable_specs,
              noise_seed))

    return cls(
        dp_sum_query=tree_aggregation_sum_query,
        num_microbatches=num_microbatches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        *args,
        **kwargs)

  return return_dpftrl_tree_aggregation_optimizer


def make_keras_optimizer_class(cls: Type[tf.keras.optimizers.Optimizer]):
  """Returns a differentially private optimizer using the `GaussianSumQuery`.

  For backwards compatibility, we create this symbol to match the previous
  output of `make_keras_optimizer_class` but using the new logic.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.keras.optimizers.Optimizer`.
  """
  warnings.warn(
      '`make_keras_optimizer_class` will be depracated on 2023-02-23. '
      'Please switch to `make_gaussian_query_optimizer_class` and the '
      'generic optimizers (`make_keras_generic_optimizer_class`).')
  return make_gaussian_query_optimizer_class(
      make_keras_generic_optimizer_class(cls))


GenericDPAdagradOptimizer = make_keras_generic_optimizer_class(
    tf.keras.optimizers.legacy.Adagrad)
GenericDPAdamOptimizer = make_keras_generic_optimizer_class(
    tf.keras.optimizers.legacy.Adam)
GenericDPSGDOptimizer = make_keras_generic_optimizer_class(
    tf.keras.optimizers.legacy.SGD)

DPFTRLTreeAggregationOptimizer = (
    make_dpftrl_tree_aggregation_optimizer_class(GenericDPSGDOptimizer))
# We keep the same names for backwards compatibility.
DPKerasAdagradOptimizer = make_gaussian_query_optimizer_class(
    GenericDPAdagradOptimizer)
DPKerasAdamOptimizer = make_gaussian_query_optimizer_class(
    GenericDPAdamOptimizer)
DPKerasSGDOptimizer = make_gaussian_query_optimizer_class(GenericDPSGDOptimizer)

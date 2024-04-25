# Copyright 2022, The TensorFlow Authors.
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
"""Computes per-example loss clip weights.

For a given Keras model and batch of inputs, computes the per-example
clip weights so that the gradient of the loss function, weighted by these
weights, is equivalent to the gradient of the original loss function but
with the per-example gradients clipped by some clip weight. Uses a variant
of the approach given in https://arxiv.org/pdf/2009.03106.pdf (see the
`compute_gradient_norms()` function).
"""

import collections
from collections.abc import Sequence
from typing import Optional

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def get_registry_generator_fn(
    tape: tf.GradientTape,
    layer_registry: lr.LayerRegistry,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
):
  """Creates the generator function for `compute_gradient_norms()`."""
  if layer_registry is None:
    # Needed for backwards compatibility.
    registry_generator_fn = None
  else:

    def registry_generator_fn(layer_instance, args, kwargs):
      if layer_instance.trainable_variables:
        # Only trainable variables factor into the gradient.
        if not layer_registry.is_elem(layer_instance):
          raise NotImplementedError(
              'Layer %s is not in the registry of known layers that can '
              'be used for efficient gradient clipping.'
              % layer_instance.__class__.__name__
          )
        registry_fn = layer_registry.lookup(layer_instance)
        (layer_vars, layer_outputs, layer_sqr_norm_fn) = registry_fn(
            layer_instance, args, kwargs, tape, num_microbatches
        )
        return layer_outputs, (
            str(id(layer_instance)),
            layer_vars,
            layer_sqr_norm_fn,
            layer_instance.trainable_weights,
        )
      else:
        # Non-trainable layer.
        return layer_instance(*args, **kwargs), None

  return registry_generator_fn


def _infer_per_example_loss_fn(model: tf.keras.Model):
  """Infer the per-example loss from model config."""

  def _convert(loss_fn):
    loss_config = loss_fn.get_config()
    loss_config['reduction'] = tf.keras.losses.Reduction.NONE
    return loss_fn.from_config(loss_config)

  model_loss = model.loss
  if isinstance(model_loss, tf.keras.losses.Loss):
    return _convert(model_loss)
  elif isinstance(model_loss, dict):
    # Note that we cannot call the public method `.get_compile_config()` because
    # it calls a numpy function, which is not supported inside a `tf.function`
    # wrapped function.
    compile_config = model._compile_config.config  # pylint: disable=protected-access
    if compile_config is None:
      raise ValueError('Model must be compiled for loss function conversion')
    # Does a weighted mean of the configured losses. Note that we cannot build
    # from the config of the compiled loss because (i) it builds a
    # `keras.metrics.Mean` class, which generates non-unique `tf.Variable`s
    # during its construction, (ii) non-unique `tf.Variables` cannot be used
    # inside a `tf.function`, which is usually where this function is used.
    if 'loss_weights' not in compile_config:
      raise ValueError(
          'Models with multiple loss must have corresponding loss weights for'
          ' loss function conversion'
      )
    weights = compile_config['loss_weights']
    per_example_losses = {k: _convert(v) for k, v in model_loss.items()}
    num_losses = len(weights)

    def _per_example_loss_fn(y_true, y_pred, sample_weight=None):
      loss_values = []
      if model_loss.keys() - y_pred.keys():
        raise ValueError(
            'y_pred must contain the same keys and the model losses, but '
            'got %s and %s' % (y_pred.keys(), model_loss.keys())
        )
      if model_loss.keys() - y_true.keys():
        raise ValueError(
            'y_true must contain the same keys and the model losses, but '
            'got %s and %s' % (y_true.keys(), model_loss.keys())
        )
      if sample_weight is not None:
        if model_loss.keys() - sample_weight.keys():
          raise ValueError(
              'sample_weight must contain the same keys and the model losses,'
              ' but got %s and %s' % (y_true.keys(), model_loss.keys())
          )
      for k in y_true.keys():
        sgl_sample_weight = None if sample_weight is None else sample_weight[k]
        sgl_value = (
            weights[k]
            * per_example_losses[k](y_true[k], y_pred[k], sgl_sample_weight)
            / num_losses
        )
        loss_values.append(tf.reshape(sgl_value, shape=[-1]))
      return tf.math.add_n(loss_values)

    return _per_example_loss_fn
  else:
    raise ValueError(
        'Unsupported type for loss function conversion: {}'.format(
            type(model_loss)
        )
    )


def compute_gradient_norms(
    input_model: tf.keras.Model,
    layer_registry: lr.LayerRegistry,
    x_batch: type_aliases.InputTensors,
    y_batch: type_aliases.OutputTensors,
    weight_batch: Optional[tf.Tensor] = None,
    per_example_loss_fn: Optional[type_aliases.LossFn] = None,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
    trainable_vars: Optional[Sequence[tf.Variable]] = None,
):
  """Computes the per-example loss gradient norms for given data.

  Applies a variant of the approach given in
    https://arxiv.org/pdf/2009.03106.pdf

  Args:
    input_model: The `tf.keras.Model` from which to obtain the layers from. The
      loss of the model *must* be a scalar loss. When using microbatching, the
      loss reduction must be mean.
    layer_registry: A `LayerRegistry` instance containing functions that help
      compute gradient norms quickly. See
      `tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry` for
      more details.
    x_batch: An `InputTensor` representing a batch of inputs to the model. The
      first axis must be the batch dimension.
    y_batch: An `OutputTensor` representing a batch of output labels. The first
      axes of the tensors must be the batch dimension. The number of examples
      should match the number of examples in `x_batch`.
    weight_batch: Optional batch of weights, passed to the loss function.
      Weights apply to the loss prior to clipping.
    per_example_loss_fn: takes as input predictions, labels and weights, and
      outputs a vector of per-example losses. If None, derived from
      `input_model.loss` by disabling its reduction.
    num_microbatches: An optional number or scalar `tf.Tensor` for the number of
      microbatches. If not None, indicates that the loss is grouped into
      num_microbatches (in this case, the batch dimension needs to be a multiple
      of num_microbatches). When there is microbatches, we always assume the
      loss is the mean over a microbatch. And the gradient norm is computed for
      each microbatch.
    trainable_vars: The list of variables included in computing the gradient
      norm. When a layer has multiple variables, we include all the variables if
      any of the variables is in the list. If `trainable_vars` is None, all the
      variables are included.

  Returns:
    A scalar vector, whose i-th entry is the norm of the gradient of the i-th
    weighted example loss (when num_microbatches is None) or the norm of the
    gradient of the i-th microbatch loss (define as a mean over the microbatch).
    Note that when the loss is weighted (`weight_batch` is not None), weights
    are applied prior to clipping.
  """
  tape = tf.GradientTape(persistent=True, watch_accessed_variables=False)
  registry_generator_fn = get_registry_generator_fn(
      tape, layer_registry, num_microbatches
  )
  # First loop computes the model outputs, summed loss, and generator outputs.
  with tape:
    model_outputs, generator_outputs_list = (
        gradient_clipping_utils.model_forward_pass(
            input_model, x_batch, generator_fn=registry_generator_fn
        )
    )

    # Ignore the original loss function's reduction to get per-example loss.
    if per_example_loss_fn is None:
      per_example_loss_fn = _infer_per_example_loss_fn(input_model)

    losses = per_example_loss_fn(y_batch, model_outputs, weight_batch)
    if losses.shape is None:
      raise NotImplementedError(
          "The unreduced (or per-example) loss's shape cannot be `None`"
      )
    if len(losses.shape) != 1:
      raise NotImplementedError(
          'The unreduced (or per-example) loss needs to have a shape of length '
          'one, but received an unreduced loss of shape length %s'
          % len(losses.shape)
      )
    if num_microbatches is not None:
      losses = tf.reduce_mean(
          common_manip_utils.maybe_add_microbatch_axis(
              losses, num_microbatches
          ),
          axis=1,
      )
    summed_loss = tf.reduce_sum(losses)
  # Unwrap the generator outputs so that the next loop avoids duplicating
  # backprop ops.
  filtered_outputs = [t for t in generator_outputs_list if t is not None]
  if trainable_vars is not None:
    # Create a set using `ref()` for fast set membership check. tf.Variable
    # itself is not hashable.
    trainable_vars = set([v.ref() for v in trainable_vars])
  layer_vars = collections.defaultdict(list)
  layer_sqr_norm_fns = collections.defaultdict(list)
  # The case of shared weights:
  #   If a layer is called k times, it will appear k times in filtered_outputs,
  #   with the same id, but potentially with different v and f. The code below
  #   groups filtered_outputs by layer_id, so we can correctly compute gradient
  #   norms. The gradient norm of a layer that occurs k times is computed as
  #   $sqrt(k * \sum_i c_i^2)$ where $c_i$ is the norm estimate of its i-th
  #   occurrence. This is an over-estimate of the actual norm. For more details,
  #   see the explanation in go/dp-sgd-shared-weights.
  for layer_id, v, f, weights_list in filtered_outputs:
    if trainable_vars is None or any(
        w.ref() in trainable_vars for w in weights_list
    ):
      layer_vars[layer_id].append(v)
      layer_sqr_norm_fns[layer_id].append(f)
  # Second loop evaluates the squared L2 norm functions and appends the results.
  layer_grad_vars = tape.gradient(
      summed_loss,
      layer_vars,
      unconnected_gradients=tf.UnconnectedGradients.ZERO,
  )
  if not layer_grad_vars:
    raise ValueError('The gradient list cannot be empty.')
  sqr_norm_list = []
  for layer_id in layer_sqr_norm_fns.keys():
    fns = layer_sqr_norm_fns[layer_id]
    grads = layer_grad_vars[layer_id]
    # Number of duplicates for this layer in `filtered_outputs`.
    num_passes = len(fns)
    if len(fns) != len(grads):
      raise ValueError(
          'There must be as many gradients as squared norm functions.'
      )
    # See go/dp-sgd-shared-weights for more details.
    for fn, grad in zip(fns, grads):
      sqr_norm_list.append(num_passes * fn(grad))
  sqr_norm_tsr = tf.stack(sqr_norm_list, axis=1)
  return tf.sqrt(tf.reduce_sum(sqr_norm_tsr, axis=1))


def compute_clip_weights(l2_norm_clip: float, gradient_norms: tf.Tensor):
  """Computes the per-example loss/clip weights for clipping.

  When the sum of the per-example losses is replaced a weighted sum, where
  the weights are generated by this method, then the gradients of each
  term in the weighted sum are clipped by the given clip value.

  Args:
    l2_norm_clip: A `float` indicating the norm to which per-example gradients
      will be clipped. That is, all gradients of the per-example loss functions
      will have norm at most `l2_norm_clip`.
    gradient_norms: A 1D `tf.Tensor` whose i-th entry is the norm of the
      gradient of the loss function for the i-th input.

  Returns:
    A 1D `tf.Tensor` representing whose i-th entry `C[i]` is either `1.0` if the
    norm of the gradient of i-th per-example loss `G[i]` is less than
    `l2_norm_clip` or a number less than `1.0` so that
    `|G[i]| * C[i] == l2_norm_clip` otherwise.
  """
  if l2_norm_clip is None:
    return None
  return l2_norm_clip / tf.math.maximum(l2_norm_clip, gradient_norms)


def compute_clipped_gradients_and_outputs(
    input_model: tf.keras.Model,
    l2_norm_clip: float,
    layer_registry: lr.LayerRegistry,
    x_batch: type_aliases.InputTensors,
    y_batch: type_aliases.OutputTensors,
    weight_batch: Optional[tf.Tensor] = None,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
    clipping_loss: Optional[type_aliases.LossFn] = None,
) -> tuple[Sequence[tf.Tensor], tf.Tensor, tf.Tensor]:
  """Computes the per-example clipped loss gradient and other useful outputs.

  Given a batch of observations `(x_batch, y_batch, weight_batch)`, the main
  steps of this function are:
  (i) compute the l2-norm of the gradients w.r.t. the trainable variables of
  `input_model`, for each weighted example loss in the batch;
  (ii) use the norms computed in (i) to obtain "clip_weights" that are used to
  reweight the loss function, such that each gradient of this reweighted loss
  has l2-norm at most `l2_norm_clip`.

  Args:
    input_model: The `tf.keras.Model` from which to obtain the layers from.
    l2_norm_clip: A `float` indicating the norm to which per-example gradients
      will be clipped. That is, all gradients of the per-example loss functions
      will have norm at most `l2_norm_clip`.
    layer_registry: A `dict` of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a `tuple` `(output, sqr_grad_norms, vars)`, where
      `output` is the pre-activator tensor, `sqr_grad_norms` is related to the
      squared norms of a layer's pre-activation tensor, and `vars` are relevant
      trainable weights (see `layer_registry_factories.py` for examples).
    x_batch: An `InputTensor` representing a batch of inputs to the model. The
      first axes of each tensor must be the batch dimension.
    y_batch: An `OutputTensor` representing a batch of output labels. The first
      axes of each tensor must be the batch dimension. The number of examples
      should match the number of examples in `x_batch`.
    weight_batch: Optional vector of weights, passed to the loss function. Must
      be of size [batch_size]. In case of microbatching, this will be reshaped
      to [num_microbatches, batch_size/num_microbatches] before passing it to
      the loss.
    num_microbatches: An optional number or scalar `tf.Tensor` for the number of
      microbatches. If not None, indicates that the loss is grouped into
      num_microbatches (in this case, the batch dimension needs to be a multiple
      of num_microbatches).
    clipping_loss: If provided, used for the clipping computation. Defaults to
      `input_model.compiled_loss`. Specifying a `clipping_loss` can be useful to
      avoid calling `input_model.compiled_loss`, as this will append the value
      of the clipped loss to the reported metrics, and this can be misleading as
      the value of the clipped loss does not reflect the true loss.

  Returns:
    clipped_grad: list of the clipped gradients of the loss function (one per
      trainable variable in `input_model`).
    y_pred: the result of applying `input_model` to `x_batch`.
    clipping_loss_value: the loss value weighted in such a way that its gradient
      is `clipped_grad`.
  """
  if hasattr(input_model.loss, 'reduction'):
    if input_model.loss.reduction == 'none':
      raise NotImplementedError(
          'Fast gradient clipping does not support '
          'models with unreduced loss functions.'
      )
  if clipping_loss is None:
    clipping_loss = input_model.compiled_loss
  gradient_norms = compute_gradient_norms(
      input_model,
      layer_registry,
      x_batch,
      y_batch,
      weight_batch,
      num_microbatches=num_microbatches,
      trainable_vars=input_model.trainable_variables,
  )
  clip_weights = compute_clip_weights(l2_norm_clip, gradient_norms)
  if weight_batch is not None:
    # Let w be the `weight_batch`, c be the `clip_weights`, and l be the losses.
    # c is computed based on the gradient of w*l, so that if we scale w*l by c,
    # the result has bounded per-example gradients. So the loss to optimize is
    # c*w*l. Here we compute c*w before passing it to the loss.
    weight_batch = common_manip_utils.maybe_add_microbatch_axis(
        weight_batch, num_microbatches
    )
    if num_microbatches is None:
      c = clip_weights  # shape [num_microbatches]
    else:
      # In this case, weight_batch is of shape [batch_size, microbatch_size],
      # we multiply by the clip_weights (which is of shape [num_microbatches])
      c = clip_weights[:, tf.newaxis]
    clip_weights = tf.nest.map_structure(lambda w: c * w, weight_batch)

  with tf.GradientTape() as tape:
    # WARNING: When num_microbatches is not None, we need to be sure that
    # `compute_loss` always computes the mean over the microbatches
    # as it is the assumption made when computing the gradient norm.
    # It is indeed the case for multiple keras loss functions
    # (e.g. mean_squared_error and binary_crossentropy). However it
    # is not defined in the contract so may not hold, especially for
    # custom losses.
    y_pred = input_model(x_batch, training=True)
    mb_y_batch = common_manip_utils.maybe_add_microbatch_axis(
        y_batch, num_microbatches
    )
    mb_y_pred = common_manip_utils.maybe_add_microbatch_axis(
        y_pred, num_microbatches
    )
    clipping_loss_value = clipping_loss(mb_y_batch, mb_y_pred, clip_weights)
  clipped_grads = tape.gradient(
      clipping_loss_value,
      input_model.trainable_variables,
      unconnected_gradients=tf.UnconnectedGradients.ZERO,
  )
  return clipped_grads, y_pred, clipping_loss_value

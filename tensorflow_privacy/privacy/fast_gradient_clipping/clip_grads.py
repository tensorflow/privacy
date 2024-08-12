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
from collections.abc import Mapping, Sequence
from typing import Optional

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def _compute_gradient_norms_internal(
    registry_fn_outputs_list: Sequence[
        gradient_clipping_utils.RegistryGeneratorFunctionOutput
    ],
    layer_grad_vars: Mapping[str, Sequence[type_aliases.Tensor]],
    trainable_vars: Optional[Sequence[tf.Variable]] = None,
) -> tf.Tensor:
  """Computes the per-example loss gradient norms for given data.

  Args:
    registry_fn_outputs_list: A sequence of RegistryGeneratorFunctionOutput
      containing information required to compute the gradient norms and
      contribution counts. Output from
      `gradient_clipping_utils.model_forward_backward_pass()`.
    layer_grad_vars: A mapping of layer id to a list of gradients for each
      trainable variable in the layer. Output from
      `gradient_clipping_utils.model_forward_backward_pass()`.
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

  Raises:
    ValueError: If `layer_grad_vars` is empty.
    ValueError: If the number of gradients for a layer is not equal to the
     number of squared norm functions for that layer.
  """
  if trainable_vars is not None:
    # Create a set using `ref()` for fast set membership check. tf.Variable
    # itself is not hashable.
    trainable_vars = set([v.ref() for v in trainable_vars])

  layer_sqr_norm_fns = collections.defaultdict(list)
  # The case of shared weights:
  #   If a layer is called k times, it will appear k times in filtered_outputs,
  #   with the same id, but potentially with different v and f. The code below
  #   groups filtered_outputs by layer_id, so we can correctly compute gradient
  #   norms. The gradient norm of a layer that occurs k times is computed as
  #   $sqrt(k * \sum_i c_i^2)$ where $c_i$ is the norm estimate of its i-th
  #   occurrence. This is an over-estimate of the actual norm. For more details,
  #   see the explanation in go/dp-sgd-shared-weights.
  for registry_fn_output in registry_fn_outputs_list:
    if trainable_vars is None or any(
        w.ref() in trainable_vars
        for w in registry_fn_output.layer_trainable_weights
    ):
      layer_sqr_norm_fns[registry_fn_output.layer_id].append(
          registry_fn_output.layer_sqr_norm_fn
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
  gradient_norms = tf.sqrt(tf.reduce_sum(sqr_norm_tsr, axis=1))
  return gradient_norms


def compute_gradient_norms(
    input_model: tf.keras.Model,
    layer_registry: lr.LayerRegistry,
    x_batch: type_aliases.InputTensors,
    y_batch: type_aliases.OutputTensors,
    weight_batch: Optional[tf.Tensor] = None,
    per_example_loss_fn: Optional[type_aliases.LossFn] = None,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
    trainable_vars: Optional[Sequence[tf.Variable]] = None,
) -> tf.Tensor:
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
  registry_generator_fn = gradient_clipping_utils.get_registry_generator_fn(
      tape=tape,
      layer_registry=layer_registry,
      sparse_noise_layer_registry=None,
      num_microbatches=num_microbatches,
  )
  layer_grad_vars, generator_outputs_list = (
      gradient_clipping_utils.model_forward_backward_pass(
          tape=tape,
          input_model=input_model,
          x_batch=x_batch,
          y_batch=y_batch,
          registry_generator_fn=registry_generator_fn,
          weight_batch=weight_batch,
          per_example_loss_fn=per_example_loss_fn,
          num_microbatches=num_microbatches,
      )
  )
  return _compute_gradient_norms_internal(
      registry_fn_outputs_list=generator_outputs_list,
      layer_grad_vars=layer_grad_vars,
      trainable_vars=trainable_vars,
  )


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
    registry_fn_outputs_list: Sequence[
        gradient_clipping_utils.RegistryGeneratorFunctionOutput
    ],
    layer_grad_vars: Mapping[str, Sequence[type_aliases.Tensor]],
    l2_norm_clip: float,
    x_batch: type_aliases.InputTensors,
    y_batch: type_aliases.OutputTensors,
    weight_batch: Optional[tf.Tensor] = None,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
    clipping_loss: Optional[type_aliases.LossFn] = None,
) -> tuple[Sequence[type_aliases.Tensor], tf.Tensor, tf.Tensor]:
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
    registry_fn_outputs_list: A `list` of RegistryGeneratorFunctionOutput
      containing information required to compute the gradient norms and
      contribution counts. Output from
      `gradient_clipping_utils.model_forward_backward_pass()`.
    layer_grad_vars: A mapping of layer id to a list of gradients for each
      trainablev ariable in the layer. Output from
      `gradient_clipping_utils.model_forward_backward_pass()`.
    l2_norm_clip: A `float` indicating the norm to which per-example gradients
      will be clipped. That is, all gradients of the per-example loss functions
      will have norm at most `l2_norm_clip`.
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
  gradient_norms = _compute_gradient_norms_internal(
      registry_fn_outputs_list=registry_fn_outputs_list,
      layer_grad_vars=layer_grad_vars,
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

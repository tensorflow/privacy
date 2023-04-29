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

from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr

InputTensor = Union[tf.Tensor, Iterable[tf.Tensor], Dict[Text, tf.Tensor]]
LossFn = Callable[..., tf.Tensor]


def get_registry_generator_fn(
    tape: tf.GradientTape,
    layer_registry: lr.LayerRegistry,
    num_microbatches: Optional[lr.BatchSize] = None,
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
            layer_vars,
            layer_sqr_norm_fn,
            layer_instance.trainable_weights,
        )
      else:
        # Non-trainable layer.
        return layer_instance(*args, **kwargs), None

  return registry_generator_fn


def compute_gradient_norms(
    input_model: tf.keras.Model,
    x_batch: InputTensor,
    y_batch: tf.Tensor,
    layer_registry: lr.LayerRegistry,
    per_example_loss_fn: Optional[LossFn] = None,
    num_microbatches: Optional[lr.BatchSize] = None,
    trainable_vars: Optional[List[tf.Variable]] = None,
):
  """Computes the per-example loss gradient norms for given data.

  Applies a variant of the approach given in
    https://arxiv.org/pdf/2009.03106.pdf

  Args:
    input_model: The `tf.keras.Model` from which to obtain the layers from. The
      loss of the model *must* be a scalar loss.
    x_batch: An `InputTensor` representing a batch of inputs to the model. The
      first axis must be the batch dimension.
    y_batch: A `tf.Tensor` representing a batch of output labels. The first axis
      must be the batch dimension. The number of examples should match the
      number of examples in `x_batch`.
    layer_registry: A `LayerRegistry` instance containing functions that help
      compute gradient norms quickly. See
      `tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry` for
      more details.
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
    A 1D `tf.Tensor` whose i-th entry is the norm of the gradient of the i-th
    per-example loss function.
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
      loss_config = input_model.loss.get_config()
      loss_config['reduction'] = tf.keras.losses.Reduction.NONE
      per_example_loss_fn = input_model.loss.from_config(loss_config)
    losses = per_example_loss_fn(y_batch, model_outputs)
    if num_microbatches is not None:
      losses = tf.reduce_mean(
          lr.add_microbatch_axis(losses, num_microbatches), axis=1
      )
    summed_loss = tf.reduce_sum(losses)
  # Unwrap the generator outputs so that the next loop avoids duplicating
  # backprop ops.
  filtered_outputs = [t for t in generator_outputs_list if t is not None]
  vars_list = []
  sqr_norm_fns_list = []
  if trainable_vars is not None:
    # Create a set using `ref()` for fast set membership check. tf.Variable
    # itself is not hashable.
    trainable_vars = set([v.ref() for v in trainable_vars])
  for v, f, weights_list in filtered_outputs:
    if trainable_vars is None or any(
        w.ref() in trainable_vars for w in weights_list
    ):
      # Include only those variables in trainable_vars.
      vars_list.append(v)
      sqr_norm_fns_list.append(f)
  # Second loop evaluates the squared L2 norm functions and appends the results.
  grads_list = tape.gradient(
      summed_loss,
      vars_list,
      unconnected_gradients=tf.UnconnectedGradients.ZERO,
  )
  sqr_norm_list = []
  for grads, f in zip(grads_list, sqr_norm_fns_list):
    sqr_norm_list.append(f(grads))
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
    x_batch: InputTensor,
    y_batch: tf.Tensor,
    l2_norm_clip: float,
    layer_registry: lr.LayerRegistry,
    num_microbatches: Optional[lr.BatchSize] = None,
    clipping_loss: Optional[LossFn] = None,
) -> Tuple[List[tf.Tensor], tf.Tensor, tf.Tensor]:
  """Computes the per-example clipped loss gradient and other useful outputs.

  Given a batch of observations `(x_batch, y_batch)`, the main steps of this
  function are: (i) compute the l2-norm of the gradients of the trainable
  variables of `input_model` for each example in the batch; (ii) use the norms
  computed in (i) to obtain "clip_weights" that are used to generate a weighted
  loss function whose gradient for each example has l2-norm at most
  `l2_norm_clip`; (iii) output the clipped gradients in (ii) and other useful
  outputs to the caller.

  Args:
    input_model: The `tf.keras.Model` from which to obtain the layers from.
    x_batch: An `InputTensor` representing a batch of inputs to the model. The
      first axis must be the batch dimension.
    y_batch: A `tf.Tensor` representing a batch of output labels. The first axis
      must be the batch dimension. The number of examples should match the
      number of examples in `x_batch`.
    l2_norm_clip: A `float` indicating the norm to which per-example gradients
      will be clipped. That is, all gradients of the per-example loss functions
      will have norm at most `l2_norm_clip`.
    layer_registry: A `dict` of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a `tuple` `(output, sqr_grad_norms, vars)`, where
      `output` is the pre-activator tensor, `sqr_grad_norms` is related to the
      squared norms of a layer's pre-activation tensor, and `vars` are relevant
      trainable weights (see `layer_registry_factories.py` for examples).
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
    A `tuple` `(grad, y_pred, clipping_loss_value)`. The first element is the
    clipped gradient of the loss function, the second is the result of
    applying `input_model` to `x_batch`, and the third is loss value of
    `input_model`, weighted by the loss weights generated by a specific
    `compute_clip_weights()` call.
  """
  if clipping_loss is None:
    clipping_loss = input_model.compiled_loss
  gradient_norms = compute_gradient_norms(
      input_model,
      x_batch,
      y_batch,
      layer_registry,
      num_microbatches=num_microbatches,
      trainable_vars=input_model.trainable_variables,
  )
  loss_weights = compute_clip_weights(l2_norm_clip, gradient_norms)
  with tf.GradientTape() as tape:
    # WARNING: When num_microbatches is not None, we need to be sure that
    # `compute_loss` always computes the mean over the microbatches
    # as it is the assumption made when computing the gradient norm.
    # It is indeed the case for multiple keras loss functions
    # (e.g. mean_squared_error and binary_crossentropy). However it
    # is not defined in the contract so may not hold, especially for
    # custom losses.
    y_pred = input_model(x_batch, training=True)
    loss_y_batch = (
        y_batch
        if num_microbatches is None
        else lr.add_microbatch_axis(y_batch, num_microbatches)
    )
    loss_y_pred = (
        y_pred
        if num_microbatches is None
        else lr.add_microbatch_axis(y_pred, num_microbatches)
    )
    clipping_loss_value = clipping_loss(loss_y_batch, loss_y_pred, loss_weights)
  clipped_grads = tape.gradient(
      clipping_loss_value,
      input_model.trainable_variables,
      unconnected_gradients=tf.UnconnectedGradients.ZERO,
  )
  return clipped_grads, y_pred, clipping_loss_value

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

from typing import Any, Callable, Dict, Iterable, Optional, Text, Union

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr

InputTensor = Union[tf.Tensor, Iterable[tf.Tensor], Dict[Text, tf.Tensor]]


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
        return layer_outputs, (layer_vars, layer_sqr_norm_fn)
      else:
        # Non-trainable layer.
        return layer_instance(*args, **kwargs), None

  return registry_generator_fn


def compute_gradient_norms(
    input_model: tf.keras.Model,
    x_batch: InputTensor,
    y_batch: tf.Tensor,
    layer_registry: lr.LayerRegistry,
    per_example_loss_fn: Optional[Callable[[tf.Tensor, Any], tf.Tensor]] = None,
    num_microbatches: Optional[lr.BatchSize] = None,
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
    per_example_loss_fn: If not None, used as the function to compute the
      vectorized per example loss. Otherwise, we derive it from `input_model`'s
      loss function.
    num_microbatches: An optional number or scalar `tf.Tensor` for the number of
      microbatches. If not None, indicates that the loss is grouped into
      num_microbatches (in this case, the batch dimension needs to be a multiple
      of num_microbatches). When there is microbatches, we always assume the
      loss is the mean over a microbatch. And the gradient norm is computed for
      each microbatch.

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
  vars_list = [a for (a, b) in filtered_outputs]
  sqr_norm_fns_list = [b for (a, b) in filtered_outputs]
  # Second loop evaluates the squared L2 norm functions and appends the results.
  grads_list = tape.gradient(summed_loss, vars_list)
  sqr_norm_list = []
  for grads, f in zip(grads_list, sqr_norm_fns_list):
    sqr_norm_list.append(f(grads))
  del tape
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


def compute_pred_and_clipped_gradients(
    input_model: tf.keras.Model,
    x_batch: InputTensor,
    y_batch: tf.Tensor,
    l2_norm_clip: float,
    layer_registry: lr.LayerRegistry,
    num_microbatches: Optional[lr.BatchSize] = None,
):
  """Computes the per-example predictions and per-example clipped loss gradient.

  Given a batch of observations `(x_batch, y_batch)`, the main steps of this
  function are: (i) compute the l2-norm of the gradients of the trainable
  variables of `input_model` for each example in the batch; (ii) use the norms
  computed in (i) to obtain "clip_weights" that are used to generate a weighted
  loss function whose gradient for each example has l2-norm at most
  `l2_norm_clip`; (iii) output the clipped gradients in (ii) and the
  `tf.Tensor` generated by `input_model` when it is given `x_batch` as its
  input.

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

  Returns:
    A `tuple` `(y_pred, grad)`. The first element is the prediction generated by
    the model on the input `x_batch`. The second element is the clipped
    gradient of the loss function.
  """
  gradient_norms = compute_gradient_norms(
      input_model,
      x_batch,
      y_batch,
      layer_registry,
      num_microbatches=num_microbatches,
  )
  loss_weights = compute_clip_weights(l2_norm_clip, gradient_norms)
  with tf.GradientTape() as tape:
    y_pred = input_model(x_batch, training=True)
    if num_microbatches is not None:
      y_batch = lr.add_microbatch_axis(y_batch, num_microbatches)
      y_pred = lr.add_microbatch_axis(y_pred, num_microbatches)
    # Warning: When num_microbatches is not None, we need to be sure that
    # `compute_loss` always computes the mean over the microbatches
    # as it is the assumption made when computing the gradient norm.
    # It is indeed the case for multiple keras loss functions
    # (e.g. mean_squared_error and binary_crossentropy). However it
    # is not defined in the contract so may not hold, especially for
    # custom losses.
    loss_value = input_model.compute_loss(
        x_batch, y_batch, y_pred, loss_weights
    )
  return y_pred, tape.gradient(loss_value, input_model.trainable_variables)

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

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils


def combine_pre_and_post_sqr_norms(pre_sqr_norm, post_grad, layer_hash):
  """Combines pre and post-activation tensors for a given variable.

  The logic for combining norms depends on the variable's underlying layer.

  Args:
    pre_sqr_norm: A `tf.Tensor` whose first dimension is the batch dimension.
      Contains squared norms that are related to the pre-activation Tensor.
    post_grad: A `tf.Tensor` whose first dimension is the batch dimension.
      Contains gradients that are related to the post-activation Tensor.
    layer_hash: A `float` that is the hash of the variable's underlying layer
      class.

  Returns:
    A 1D `tf.Tensor` whose i-th entry is the norm of the gradient of the i-th
    per-example loss function with respect to the given variable.
  """
  post_sqr_grads = tf.square(post_grad)
  if layer_hash == hash(tf.keras.layers.Embedding):
    scaled_grads = tf.expand_dims(pre_sqr_norm, axis=-1) * post_sqr_grads
    reduction_axes = tf.range(1, tf.rank(scaled_grads))
    return tf.reduce_sum(scaled_grads, axis=reduction_axes)
  else:
    reduction_axes = tf.range(1, tf.rank(post_sqr_grads))
    post_sqr_norm = tf.reduce_sum(post_sqr_grads, axis=reduction_axes)
    return pre_sqr_norm * post_sqr_norm


def compute_gradient_norms(input_model, x_batch, y_batch, layer_registry):
  """Computes the per-example loss gradient norms for given data.

  Applies the approach given in https://arxiv.org/pdf/2009.03106.pdf, except
  the batch matrix multiplication operation in Algorithm 2 is replaced with
  the computation of two norm computations.

  Args:
    input_model: The `tf.keras.Model` from which to obtain the layers from. The
      loss of the model *must* be a scalar loss.
    x_batch: A `tf.Tensor` representing a batch of inputs to the model. The
      first axis must be the batch dimension.
    y_batch: A `tf.Tensor` representing a batch of output labels. The first axis
      must be the batch dimension. The number of examples should match the
      number of examples in `x_batch`.
    layer_registry: A `dict` of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a `tuple` `(output, sqr_grad_norms, vars)`, where
      `output` is the pre-activator tensor, `sqr_grad_norms` is related to the
      squared norms of a layer's pre-activation tensor, and `vars` are relevant
      trainable weights (see `layer_registry_factories.py` for examples).

  Returns:
    A 1D `tf.Tensor` whose i-th entry is the norm of the gradient of the i-th
    per-example loss function.
  """
  tape = tf.GradientTape(persistent=True, watch_accessed_variables=False)
  # First loop computes the norms of the layer inputs, caches these inputs,
  # and computes the summed loss.
  with tape:
    model_outputs, pre_norm_list, var_list, layer_hash_list = (
        gradient_clipping_utils.forward_norm_pass(
            input_model, x_batch, tape, layer_registry
        )
    )
    # Ignore the original loss function's reduction to get per-example loss.
    loss_config = input_model.loss.get_config()
    loss_config['reduction'] = tf.keras.losses.Reduction.NONE
    per_example_loss_fn = input_model.loss.from_config(loss_config)
    losses = per_example_loss_fn(y_batch, model_outputs)
    if tf.rank(tf.squeeze(losses)) > 1:
      raise NotImplementedError('Vector losses are not supported.')
    summed_loss = tf.reduce_sum(losses)
  # Second loop computes the norm of the gradient of the loss with respect to
  # the pre-activation tensors, and multiplies these norms with the results of
  # the first loop.
  full_norm_list = []
  grads = tape.gradient(summed_loss, var_list)
  for i in range(len(var_list)):
    full_norm = combine_pre_and_post_sqr_norms(
        pre_norm_list[i], grads[i], layer_hash_list[i]
    )
    full_norm_list.append(full_norm)
  del tape
  # Post-processing for compatibility with non-eager mode (very annoying).
  full_norm_tsr = tf.stack(full_norm_list, axis=1)
  return tf.sqrt(tf.reduce_sum(full_norm_tsr, axis=1))


def compute_clip_weights(l2_norm_clip, gradient_norms):
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
    input_model, x_batch, y_batch, l2_norm_clip, layer_registry
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
    x_batch: A `tf.Tensor` representing a batch of inputs to the model. The
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

  Returns:
    A `tuple` `(y_pred, grad)`. The first element is the prediction generated by
    the model on the input `x_batch`. The second element is the clipped
    gradient of the loss function.
  """
  gradient_norms = compute_gradient_norms(
      input_model, x_batch, y_batch, layer_registry
  )
  loss_weights = compute_clip_weights(l2_norm_clip, gradient_norms)
  with tf.GradientTape() as tape:
    y_pred = input_model(x_batch, training=True)
    loss_value = input_model.compute_loss(
        x_batch, y_batch, y_pred, loss_weights
    )
  return y_pred, tape.gradient(loss_value, input_model.trainable_variables)

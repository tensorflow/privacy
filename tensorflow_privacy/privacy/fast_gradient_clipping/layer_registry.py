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
"""Defines the layer registry class and useful factory functions.

Defines "fast" gradient norm layer registry functions for use in the "fast"
gradient clipping algorithm. Specifically, each registry function takes
in two inputs (i) a layer instance and (ii) `tf.Tensor` inputs to produce three
outputs: (a) a differentiable `tf.Tensor` `Z`, (b) either `None` or a function
that maps the object in (a) to the layer instance's output when using the
inputs in (ii), and (c) a function `F` that generates the per-example
squared gradient norms when it is fed an object representing the gradient of
the summed loss with respect to `Z` in (a). If (b) is `None`, then (a) is
expected to contain the layer outputs.

When a layer registry function is defined, it is generally assumed that the
following relation holds:

  `|dL/dW|^2 == F(grad_Z)`

where `gradient_Z` is the gradient of the summed loss with respect to `Z`.

For example, if the layer instance is tf.keras.layers.Dense, Z contains the
pre-activation tensors, i.e., `z = X * w` for input `X`, and `g` is a tensor
whose i-th entry is the L2 norm of the i-th input vector, then

  `F(grad_Z) = g^2 * l2_row_norm(grad_Z)^2`,

where `l2_row_norm(y)` computes the L2 norm for each row of an input `y`.
Details of this decomposition can be found in https://arxiv.org/abs/1510.01799
"""

import tensorflow as tf


# ==============================================================================
# Main class
# ==============================================================================
class LayerRegistry:
  """Custom container for layer registry functions."""

  def __init__(self):
    """Basic initialization of various internal dictionaries."""
    self._layer_class_dict = {}
    self._registry = {}

  def is_elem(self, layer_instance):
    """Checks if a layer instance's class is in the registry."""
    return hash(layer_instance.__class__) in self._registry

  def lookup(self, layer_instance):
    """Returns the layer registry function for a given layer instance."""
    return self._registry[hash(layer_instance.__class__)]

  def insert(self, layer_class, layer_registry_function):
    """Inserts a layer registry function into the internal dictionaries."""
    layer_key = hash(layer_class)
    self._layer_class_dict[layer_key] = layer_class
    self._registry[layer_key] = layer_registry_function


# ==============================================================================
# Supported Keras layers
# ==============================================================================
def dense_layer_computation(layer_instance, inputs):
  """Registry function for `tf.keras.layers.Dense`.

  The logic for this computation is based on the following paper:
    https://arxiv.org/abs/1510.01799

  For the sake of efficiency, we fuse the variables and square grad norms
  for the kernel weights and bias vector together.

  Args:
    layer_instance: A `tf.keras.layers.Dense` instance.
    inputs: A `tf.Tensor` which can be passed into the layer instance, i.e.,
      `layer_instance(inputs)` returns a valid output.

  Returns:
    A `tuple` `(base_vars, transform, sqr_norm_fn)`. `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick,
    `transform` is a function that maps `base_vars` to the layer outputs, and
    `sqr_norm_fn` is a function that takes one input, a `tf.Tensor` that
    represents the output of the call `tape.gradient(summed_loss, base_vars)`
    where `tape` is a `tf.GradientTape` instance that records the dense
    layer computation and `summed_loss` is the sum of the per-example losses
    of the underlying model. This function then returns the per-example squared
    L2 gradient norms of the trainable variables in `layer_instance`. These
    squared norms should be a 1D `tf.Tensor` of length `batch_size`.
  """
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*inputs)
  layer_instance.activation = orig_activation
  def sqr_norm_fn(base_vars_grads):
    sqr_inputs = tf.square(*inputs)
    inputs_reduction_axes = tf.range(1, tf.rank(sqr_inputs))
    input_sqr_norms = tf.reduce_sum(sqr_inputs, axis=inputs_reduction_axes)
    if layer_instance.use_bias:
      # Adding a bias term is equivalent to a layer with no bias term and which
      # adds an additional variable to the layer input that only takes a
      # constant value of 1.0. This is thus equivalent to adding 1.0 to the sum
      # of the squared values of the inputs.
      input_sqr_norms += tf.cast(1.0, dtype=input_sqr_norms.dtype)
    reduction_axes = tf.range(1, tf.rank(base_vars_grads))
    base_vars_sqr_norms = tf.reduce_sum(
        tf.square(base_vars_grads), axis=reduction_axes
    )
    return input_sqr_norms * base_vars_sqr_norms

  return base_vars, layer_instance.activation, sqr_norm_fn


def embedding_layer_computation(layer_instance, inputs):
  """Registry function for `tf.keras.layers.Embedding`.

  The logic of this computation is based on the `tf.keras.layers.Dense`
  computation and the fact that an embedding layer is just a dense layer
  with no activation function and an output vector of the form X*W for input
  X, where the i-th row of W is the i-th embedding vector and the j-th row of
  X is a one-hot vector representing the input of example j.

  Args:
    layer_instance: A `tf.keras.layers.Embedding` instance.
    inputs: A `tf.Tensor` which can be passed into the layer instance, i.e.,
      `layer_instance(inputs)` returns a valid output.

  Returns:
    A `tuple` `(base_vars, None, sqr_norm_fn)`, `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick, and
    `sqr_norm_fn` is a function that takes one input, a `tf.Tensor` that
    represents the output of the call `tape.gradient(summed_loss, base_vars)`
    where `tape` is a `tf.GradientTape` instance that records the dense
    layer computation and `summed_loss` is the sum of the per-example losses
    of the underlying model. This function then returns the per-example squared
    L2 gradient norms of the trainable variables in `layer_instance`. These
    squared norms should be a 1D `tf.Tensor` of length `batch_size`.
  """
  if hasattr(layer_instance, "sparse"):  # for backwards compatibility
    if layer_instance.sparse:
      raise NotImplementedError("Sparse output vectors are not supported.")
  if len(inputs[0].shape) != 2:
    raise NotImplementedError("Only 2D embedding inputs are supported.")
  # The logic below is applied to properly handle repeated embedding indices.
  # Specifically, sqr_grad_norms will contain the total counts of each embedding
  # index (see how it is processed in the combine_pre_and_post_sqr_norms()
  # function in clip_grads.py). An example is as follows:
  #
  #   inputs =
  #     [[0 0 0 1 2 2],
  #      [0 2 2 2 1 1]]
  #
  #   counts =
  #     [[3 1 2]
  #      [1 2 3]]
  #
  #   input_counts =
  #     [[3 3 3 1 2 2],
  #      [1 3 3 3 2 2]]
  #
  base_vars = layer_instance(*inputs)
  def sqr_norm_fn(base_vars_grads):
    indices = tf.cast(*inputs, tf.int32)
    if isinstance(indices, tf.SparseTensor):
      indices = tf.sparse.to_dense(indices)
    counts = tf.math.bincount(indices, axis=-1)
    input_counts = tf.expand_dims(
        tf.cast(tf.gather(counts, indices, batch_dims=1), base_vars.dtype),
        axis=-1,
    )
    scaled_grads = input_counts * tf.square(base_vars_grads)
    reduction_axes = tf.range(1, tf.rank(scaled_grads))
    return tf.reduce_sum(scaled_grads, axis=reduction_axes)

  return base_vars, None, sqr_norm_fn


# ==============================================================================
# Main factory methods
# ==============================================================================
def make_default_layer_registry():
  registry = LayerRegistry()
  registry.insert(tf.keras.layers.Dense, dense_layer_computation)
  registry.insert(tf.keras.layers.Embedding, embedding_layer_computation)
  return registry

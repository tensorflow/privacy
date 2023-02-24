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
def dense_layer_computation(layer_instance, inputs, tape):
  """Registry function for `tf.keras.layers.Dense`.

  The logic for this computation is based on the following paper:
    https://arxiv.org/abs/1510.01799

  For the sake of efficiency, we fuse the variables and square grad norms
  for the kernel weights and bias vector together.

  Args:
    layer_instance: A `tf.keras.layers.Dense` instance.
    inputs: A `tf.Tensor` which can be passed into the layer instance, i.e.,
      `layer_instance(inputs)` returns a valid output.
    tape: A `tf.GradientTape` instance that will be used to watch the output
      `base_vars`.

  Returns:
    A `tuple` `(base_vars, outputs, sqr_norm_fn)`. `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick,
    `outputs` is the result of `layer_instance(*inputs)`, and `sqr_norm_fn` is
    a function that takes one input, a `tf.Tensor` that represents the output
    of the call `tape.gradient(summed_loss, base_vars)` where `tape` is a
    `tf.GradientTape` instance that records the dense layer computation and
    `summed_loss` is the sum of the per-example losses of the underlying model.
    This function then returns the per-example squared L2 gradient norms of the
    trainable variables in `layer_instance`. These squared norms should be a 1D
    `tf.Tensor` of length `batch_size`.
  """
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*inputs)
  tape.watch(base_vars)
  layer_instance.activation = orig_activation
  outputs = orig_activation(base_vars) if orig_activation else base_vars
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

  return base_vars, outputs, sqr_norm_fn


def embedding_layer_computation(layer_instance, inputs, tape):
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
    tape: A `tf.GradientTape` instance that will be used to watch the output
      `base_vars`.

  Returns:
    A `tuple` `(base_vars, outputs, sqr_norm_fn)`. `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick,
    `outputs` is the result of `layer_instance(*inputs)`, and `sqr_norm_fn` is
    a function that takes one input, a `tf.Tensor` that represents the output
    of the call `tape.gradient(summed_loss, base_vars)` where `tape` is a
    `tf.GradientTape` instance that records the dense layer computation and
    `summed_loss` is the sum of the per-example losses of the underlying model.
    This function then returns the per-example squared L2 gradient norms of the
    trainable variables in `layer_instance`. These squared norms should be a 1D
    `tf.Tensor` of length `batch_size`.
  """
  if hasattr(layer_instance, "sparse"):  # for backwards compatibility
    if layer_instance.sparse:
      raise NotImplementedError("Sparse output tensors are not supported.")
  if isinstance(inputs, tf.SparseTensor):
    raise NotImplementedError("Sparse input tensors are not supported.")

  # Disable experimental features.
  if hasattr(layer_instance, "_use_one_hot_matmul"):
    if layer_instance._use_one_hot_matmul:  # pylint: disable=protected-access
      raise NotImplementedError(
          "The experimental embedding feature"
          "'_use_one_hot_matmul' is not supported."
      )
  input_ids = tf.cast(*inputs, tf.int32)
  base_vars = layer_instance.trainable_variables[0]
  tape.watch(base_vars)
  outputs = tf.nn.embedding_lookup(base_vars, input_ids)

  def sqr_norm_fn(base_vars_grads):
    # Get a 1D tensor of the row indices.
    nrows = tf.shape(input_ids)[0]
    if isinstance(input_ids, tf.RaggedTensor):
      row_indices = tf.expand_dims(
          input_ids.merge_dims(1, -1).value_rowids(), axis=-1
      )
    elif isinstance(input_ids, tf.Tensor):
      ncols = tf.reduce_prod(tf.shape(input_ids)[1:])
      repeats = tf.repeat(ncols, nrows)
      row_indices = tf.reshape(tf.repeat(tf.range(nrows), repeats), [-1, 1])
    else:
      raise NotImplementedError(
          "Cannot parse input_ids of type %s" % input_ids.__class__.__name__
      )
    # Sum-reduce the `IndexSlices` that is the result of a `tape.gradient()`
    # call. The sum is reduced by the repeated embedding indices and batch
    # index. It is adapted from the logic in:
    #   tf.keras.optimizers.legacy.optimizer_v2._deduplicate_indexed_slices
    if not isinstance(base_vars_grads, tf.IndexedSlices):
      raise NotImplementedError(
          "Cannot parse embedding gradients of type: %s"
          % base_vars_grads.__class__.__name__
      )
    slice_indices = tf.expand_dims(base_vars_grads.indices, axis=-1)
    paired_indices = tf.concat(
        [tf.cast(row_indices, tf.int64), tf.cast(slice_indices, tf.int64)],
        axis=1,
    )
    (unique_paired_indices, new_index_positions) = tf.raw_ops.UniqueV2(
        x=paired_indices, axis=[0]
    )
    unique_batch_ids = unique_paired_indices[:, 0]
    summed_gradients = tf.math.unsorted_segment_sum(
        base_vars_grads.values,
        new_index_positions,
        tf.shape(unique_paired_indices)[0],
    )
    # Compute the squared gradient norms at the per-example level.
    sqr_gradient_sum = tf.reduce_sum(tf.square(summed_gradients), axis=1)
    summed_data_range = tf.range(tf.shape(sqr_gradient_sum)[0])
    return tf.sparse.segment_sum(
        sqr_gradient_sum,
        summed_data_range,
        tf.sort(unique_batch_ids),
        num_segments=nrows,
    )  # fill in empty inputs

  return base_vars, outputs, sqr_norm_fn


# ==============================================================================
# Main factory methods
# ==============================================================================
def make_default_layer_registry():
  registry = LayerRegistry()
  registry.insert(tf.keras.layers.Dense, dense_layer_computation)
  registry.insert(tf.keras.layers.Embedding, embedding_layer_computation)
  return registry

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

We also extend fast gradient norm computation to the case when the losses
are microbatched, i.e. each per example loss is the mean of a set of losses.
This could be useful for achieving user-level privacy and for improving the
quality of DP models, through better estimation of the gradients due to
aggregation at the microbatch level.
"""
# copybara.strip_begin
# The detailed algorithm can be found in go/fast-dpsgd-mb.
# copybara.strip_end

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Type, Union
import tensorflow as tf


# ==============================================================================
# Type aliases
# ==============================================================================
InputTensor = Union[tf.Tensor, Iterable[tf.Tensor], Dict[Text, tf.Tensor]]

OutputTensor = Union[tf.Tensor, Iterable[tf.Tensor]]

BatchSize = Union[int, tf.Tensor]

SquareNormFunction = Callable[[OutputTensor], tf.Tensor]

RegistryFunctionOutput = Tuple[Any, OutputTensor, SquareNormFunction]

RegistryFunction = Callable[
    [Any, Tuple[Any, ...], Dict[Text, Any], tf.GradientTape],
    RegistryFunctionOutput,
]


# ==============================================================================
# Main class
# ==============================================================================
class LayerRegistry:
  """Custom container for layer registry functions."""

  def __init__(self):
    """Basic initialization of various internal dictionaries."""
    self._layer_class_dict = {}
    self._registry = {}

  def is_elem(self, layer_instance: tf.keras.layers.Layer) -> bool:
    """Checks if a layer instance's class is in the registry."""
    return hash(layer_instance.__class__) in self._registry

  def lookup(self, layer_instance: tf.keras.layers.Layer) -> RegistryFunction:
    """Returns the layer registry function for a given layer instance."""
    return self._registry[hash(layer_instance.__class__)]

  def insert(
      self,
      layer_class: Type[tf.keras.layers.Layer],
      layer_registry_function: RegistryFunction,
  ):
    """Inserts a layer registry function into the internal dictionaries."""
    layer_key = hash(layer_class)
    self._layer_class_dict[layer_key] = layer_class
    self._registry[layer_key] = layer_registry_function


# ==============================================================================
# Utilities
# ==============================================================================
def maybe_add_microbatch_axis(
    x: tf.Tensor,
    num_microbatches: Optional[BatchSize],
) -> tf.Tensor:
  """Adds the microbatch axis.

  Args:
    x: the input tensor.
    num_microbatches: If None, x is returned unchanged. Otherwise, must divide
      the batch size.

  Returns:
    The input tensor x, reshaped from [batch_size, ...] to
    [num_microbatches, batch_size / num_microbatches, ...].
  """
  if num_microbatches is None:
    return x
  with tf.control_dependencies(
      [tf.assert_equal(tf.math.floormod(tf.shape(x)[0], num_microbatches), 0)]
  ):
    return tf.reshape(
        x, tf.concat([[num_microbatches, -1], tf.shape(x)[1:]], axis=0)
    )


# ==============================================================================
# Supported Keras layers
# ==============================================================================
def dense_layer_computation(
    layer_instance: tf.keras.layers.Dense,
    input_args: Tuple[Any, ...],
    input_kwargs: Dict[Text, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.Dense`.

  The logic for this computation is based on the following paper:
    https://arxiv.org/abs/1510.01799

  For the sake of efficiency, we fuse the variables and square grad norms
  for the kernel weights and bias vector together.

  Args:
    layer_instance: A `tf.keras.layers.Dense` instance.
    input_args: A `tuple` containing the first part of `layer_instance` input.
      Specifically, `layer_instance(*inputs_args, **input_kwargs)` should return
      a valid output.
    input_kwargs: A `tuple` containing the second part of `layer_instance`
      input. Specifically, `layer_instance(*inputs_args, **input_kwargs)` should
      return a valid output.
    tape: A `tf.GradientTape` instance that will be used to watch the output
      `base_vars`.
    num_microbatches: An optional numeric value or scalar `tf.Tensor` for
      indicating whether and how the losses are grouped into microbatches. If
      not None, num_microbatches must divide the batch size.

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
  if input_kwargs:
    raise ValueError("Dense layer calls should not receive kwargs.")
  del input_kwargs  # Unused in dense layer calls.
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*input_args)
  tape.watch(base_vars)
  layer_instance.activation = orig_activation
  outputs = orig_activation(base_vars) if orig_activation else base_vars

  def sqr_norm_fn(base_vars_grads):

    def _compute_gramian(x):
      if num_microbatches is not None:
        x_microbatched = maybe_add_microbatch_axis(x, num_microbatches)
        return tf.matmul(x_microbatched, x_microbatched, transpose_b=True)
      else:
        # Special handling for better efficiency
        return tf.reduce_sum(tf.square(x), axis=tf.range(1, tf.rank(x)))

    inputs_gram = _compute_gramian(*input_args)
    base_vars_grads_gram = _compute_gramian(base_vars_grads)
    if layer_instance.use_bias:
      # Adding a bias term is equivalent to a layer with no bias term and which
      # adds an additional variable to the layer input that only takes a
      # constant value of 1.0. This is thus equivalent to adding 1.0 to the sum
      # of the squared values of the inputs.
      inputs_gram += 1.0
    return tf.reduce_sum(
        inputs_gram * base_vars_grads_gram,
        axis=tf.range(1, tf.rank(inputs_gram)),
    )

  return base_vars, outputs, sqr_norm_fn


def embedding_layer_computation(
    layer_instance: tf.keras.layers.Embedding,
    input_args: Tuple[Any, ...],
    input_kwargs: Dict[Text, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.Embedding`.

  The logic of this computation is based on the `tf.keras.layers.Dense`
  computation and the fact that an embedding layer is just a dense layer
  with no activation function and an output vector of the form X*W for input
  X, where the i-th row of W is the i-th embedding vector and the j-th row of
  X is a one-hot vector representing the input of example j.

  Args:
    layer_instance: A `tf.keras.layers.Embedding` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  if input_kwargs:
    raise ValueError("Embedding layer calls should not receive kwargs.")
  del input_kwargs  # Unused in embedding layer calls.
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  if hasattr(layer_instance, "sparse"):  # for backwards compatibility
    if layer_instance.sparse:
      raise NotImplementedError("Sparse output tensors are not supported.")
  if isinstance(input_args[0], tf.SparseTensor):
    raise NotImplementedError("Sparse input tensors are not supported.")

  # Disable experimental features.
  if hasattr(layer_instance, "_use_one_hot_matmul"):
    if layer_instance._use_one_hot_matmul:  # pylint: disable=protected-access
      raise NotImplementedError(
          "The experimental embedding feature"
          "'_use_one_hot_matmul' is not supported."
      )
  input_ids = tf.cast(*input_args, tf.int32)
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
    row_indices = tf.cast(row_indices, tf.int32)
    if num_microbatches is not None:
      microbatch_size = tf.cast(nrows / num_microbatches, tf.int32)
      nrows = num_microbatches
      row_indices = tf.cast(
          tf.math.floordiv(row_indices, microbatch_size), tf.int32
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
def make_default_layer_registry() -> LayerRegistry:
  registry = LayerRegistry()
  registry.insert(tf.keras.layers.Dense, dense_layer_computation)
  registry.insert(tf.keras.layers.Embedding, embedding_layer_computation)
  return registry

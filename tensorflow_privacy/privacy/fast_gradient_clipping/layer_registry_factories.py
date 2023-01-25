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
"""Generates a default layer registry.

Defines "fast" gradient norm layer registry functions for use in the "fast"
gradient clipping algorithm. Specifically, each registry function takes
in two inputs (i) a layer instance and (ii) `tf.Tensor` inputs to produce three
outputs: (a) a `tf.Tensor` `G` of gradient norms, (b) a differentiable
`tf.Tensor` `Z`, and (c) either `None` or a function that maps the object in
(b) to the layer instance's output when using the inputs in (ii). If (c) is
`None`, then (b) contains the layer outputs.

When a layer registry function is defined, it is generally assumed that the
following relation holds for each pair `(g, z)` in `zip(G, Z)`:

  `|dL/dw|^2 == |dL/dz|^2 * g^2`

where `L` is any per-example loss function and `w` are the trainable variables
corresponding to `(g, z)`.

For example, this relation holds if the layer instance is tf.keras.layers.Dense,
Z contains the pre-activation tensors, i.e., `z = X * w` for input `X`, and `g`
is the norm of the input corresponding to the given per-example loss (see the
formulae in https://arxiv.org/abs/1510.01799 for more details).

The registry functions are registered in a `dict` (registry) whose key is the
hash of the layer class and whose value is the registry function.
"""

import tensorflow as tf


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
    A `tuple` `(sqr_grad_norms, base_vars, transform)`, where `norms` is a 1D
    `tf.Tensor` of the squared l2-norms of the input tensors, `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick, and
    `transform` is a function that maps `base_vars` to the layer outputs.
  """
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*inputs)
  sqr_inputs = tf.square(*inputs)
  inputs_reduction_axes = tf.range(1, tf.rank(sqr_inputs))
  sqr_grad_norms = tf.reduce_sum(tf.square(*inputs), axis=inputs_reduction_axes)
  if layer_instance.use_bias:
    # Adding a bias term is equivalent to a layer with no bias term and which
    # adds an additional variable to the layer input that only takes a constant
    # value of 1.0. This is thus equivalent to adding 1.0 to the sum of the
    # squared values of the inputs.
    sqr_grad_norms += tf.cast(1.0, dtype=sqr_grad_norms.dtype)
  layer_instance.activation = orig_activation
  return sqr_grad_norms, base_vars, layer_instance.activation


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
    A `tuple` `(sqr_grad_norms, base_vars, None)`, where `sqr_grad_norms` is
    a `tf.Tensor` that is related to the squared l2-norms of the input tensors
    and `base_vars` is the intermediate Tensor used in the chain-rule / "fast"
    clipping trick.
  """
  if layer_instance.sparse:
    raise NotImplementedError("Sparse output vectors are not supported.")
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
  #   sqr_grad_norms =
  #     [[3 3 3 1 2 2],
  #      [1 3 3 3 2 2]]
  #
  base_vars = layer_instance(*inputs)
  indices = tf.cast(*inputs, tf.int32)
  if isinstance(indices, tf.SparseTensor):
    indices = tf.sparse.to_dense(indices)
  counts = tf.math.bincount(indices, axis=-1)
  sqr_grad_norms = tf.cast(
      tf.gather(counts, indices, batch_dims=1), base_vars.dtype
  )
  return sqr_grad_norms, base_vars, None


# ==============================================================================
# Main factory methods
# ==============================================================================
def make_default_layer_registry():
  registry = {}
  registry[hash(tf.keras.layers.Dense)] = dense_layer_computation
  registry[hash(tf.keras.layers.Embedding)] = embedding_layer_computation
  return registry

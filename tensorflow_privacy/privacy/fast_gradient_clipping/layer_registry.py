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

from typing import Type

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions.dense import dense_layer_computation
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions.embedding import embedding_layer_computation


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

  def lookup(
      self, layer_instance: tf.keras.layers.Layer
  ) -> type_aliases.RegistryFunction:
    """Returns the layer registry function for a given layer instance."""
    return self._registry[hash(layer_instance.__class__)]

  def insert(
      self,
      layer_class: Type[tf.keras.layers.Layer],
      layer_registry_function: type_aliases.RegistryFunction,
  ):
    """Inserts a layer registry function into the internal dictionaries."""
    layer_key = hash(layer_class)
    self._layer_class_dict[layer_key] = layer_class
    self._registry[layer_key] = layer_registry_function


# ==============================================================================
# Main factory methods
# ==============================================================================
def make_default_layer_registry() -> LayerRegistry:
  registry = LayerRegistry()
  registry.insert(tf.keras.layers.Dense, dense_layer_computation)
  registry.insert(tf.keras.layers.Embedding, embedding_layer_computation)
  return registry

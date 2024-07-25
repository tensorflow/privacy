# Copyright 2024, The TensorFlow Authors.
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
"""Registry of layer classes to their contribution histogram functions."""

from typing import Type

import tensorflow as tf
from tensorflow_privacy.privacy.sparsity_preserving_noise import type_aliases
from tensorflow_privacy.privacy.sparsity_preserving_noise.registry_functions import embedding


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
  ) -> type_aliases.SparsityPreservingNoiseLayerRegistryFunction:
    """Returns the layer registry function for a given layer instance."""
    return self._registry[hash(layer_instance.__class__)]

  def insert(
      self,
      layer_class: Type[tf.keras.layers.Layer],
      layer_registry_function: type_aliases.SparsityPreservingNoiseLayerRegistryFunction,
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
  registry.insert(
      tf.keras.layers.Embedding,
      embedding.embedding_layer_contribution_histogram,
  )
  return registry

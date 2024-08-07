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
"""Utility functions that help in the computation of per-example gradient norms."""

from collections.abc import Callable, Sequence, Set
import dataclasses
from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


@dataclasses.dataclass(frozen=True)
class RegistryGeneratorFunctionOutput:
  layer_id: str
  layer_vars: Optional[Sequence[tf.Variable]]
  layer_sqr_norm_fn: Optional[type_aliases.SquareNormFunction]
  layer_trainable_weights: Optional[Sequence[tf.Variable]]


def has_internal_compute_graph(input_object: Any):
  """Checks if input is a TF model and has a TF internal compute graph."""
  return (
      isinstance(input_object, tf.keras.Model)
      and hasattr(input_object, '_flatten_to_reference_inputs')
      and hasattr(input_object, '_tensor_usage_count')
      and hasattr(input_object, '_conform_to_reference_input')
      and hasattr(input_object, '_nodes_by_depth')
  )


def get_registry_generator_fn(
    tape: tf.GradientTape,
    layer_registry: lr.LayerRegistry,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
) -> Optional[Callable[..., Tuple[tf.Tensor, RegistryGeneratorFunctionOutput]]]:
  """Creates the generator function for `model_forward_backward_pass()`.

  Args:
    tape: The `tf.GradientTape` to use for the gradient computation.
    layer_registry: A `dict` of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a `tuple` `(output, sqr_grad_norms, vars)`, where
      `output` is the pre-activator tensor, `sqr_grad_norms` is related to the
      squared norms of a layer's pre-activation tensor, and `vars` are relevant
      trainable
    num_microbatches: An optional number or scalar `tf.Tensor` for the number of
      microbatches. If not None, indicates that the loss is grouped into
      num_microbatches (in this case, the batch dimension needs to be a multiple
      of num_microbatches).

  Returns:
    A function that returns a `tuple` `(output, sqr_grad_norms, vars)`, where
    `output` is the pre-activator tensor, `sqr_grad_norms` is related to the
    squared norms of a layer's pre-activation tensor, and `vars` are relevant
    trainable variables.
  """
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
        return layer_outputs, RegistryGeneratorFunctionOutput(
            layer_id=str(id(layer_instance)),
            layer_vars=layer_vars,
            layer_sqr_norm_fn=layer_sqr_norm_fn,
            layer_trainable_weights=layer_instance.trainable_weights,
        )
      else:
        # Non-trainable layer.
        return layer_instance(*args, **kwargs), None

  return registry_generator_fn


def model_forward_pass(
    input_model: tf.keras.Model,
    inputs: type_aliases.PackedTensors,
    generator_fn: type_aliases.GeneratorFunction = None,
) -> tuple[type_aliases.PackedTensors, Sequence[Any]]:
  """Does a forward pass of a model and returns useful intermediates.

  NOTE: the graph traversal algorithm is an adaptation of the logic in the
    _run_internal_graph() method in the functional.Functional class. Hence,
    forward_norm_pass should only be invoked if the generated model
    instance is an instance of the functional.Functional class.

  Args:
    input_model: A `tf.keras.Model` to compute the quantities for.
    inputs: Arbitrary input to be fed into the input layer of the model. It is
      expected that `input_model(inputs)` returns a valid output.
    generator_fn: A function with signature `(tf.keras.layers.Layer, Any, Any)
      -> (tf.Tensor, Any)`, where we require `generator_fn(layer_instance, args,
      kwargs)[0] == layer_instance(*args, **kwargs)`. If `None`, then
      `layer_fn(layer_instance, args, kwargs)[1] == None`.

  Returns:
    A `tuple` `(outputs, generator_outputs_list)`. `outputs` is the
    `PackedTensor` that is generated as a result of a forward pass.
    `generator_outputs_list` is a `list` whose i-th entry is the output of
    `generator_fn(lyr, args, kwargs)[1]` where `lyr` is the i-th
    layer when the compute graph of `input_model` is traversed in BFS order.
  """
  # TODO: Avoid or remove the references to protected methods of `input_model`.  # pylint: disable=g-bad-todo

  # Default generator.
  generator_outputs_list = []
  if generator_fn is None:

    def generator_fn(layer_instance, args, kwargs):
      return layer_instance(*args, **kwargs), None

  # Prepare the inputs and BFS variables.
  flattened_inputs = input_model._flatten_to_reference_inputs(inputs)  # pylint: disable=protected-access
  tensor_dict = {}
  tensor_usage_count = input_model._tensor_usage_count  # pylint: disable=protected-access
  for x, y in zip(input_model.inputs, flattened_inputs):
    y = input_model._conform_to_reference_input(y, ref_input=x)  # pylint: disable=protected-access
    x_id = str(id(x))
    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
  nodes_by_depth = input_model._nodes_by_depth  # pylint: disable=protected-access
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Perform BFS feedforward computations.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      if node.is_input:
        continue  # inputs already exist
      if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
        continue  # node is not computable; try skipping
      args, kwargs = node.map_arguments(tensor_dict)
      if has_internal_compute_graph(node.layer):
        # If this node has an internal computational graph, we can recurse.
        node_layer_outputs, node_generator_outputs = model_forward_pass(
            node.layer, args, generator_fn
        )
        generator_outputs_list.extend(node_generator_outputs)
      else:
        # Otherwise, we parse the node directly.
        node_layer_outputs, layer_generator_outputs = generator_fn(
            node.layer, args, kwargs
        )
        generator_outputs_list.append(layer_generator_outputs)

      # Update the current dictionary of inputs for the next node.
      for x_id, y in zip(
          node.flat_output_ids, tf.nest.flatten(node_layer_outputs)
      ):
        tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

  # Gather outputs (in case there are multiple) and return.
  output_tensors = []
  for x in input_model.outputs:
    x_id = str(id(x))
    output_tensors.append(tensor_dict[x_id].pop())
  model_outputs = tf.nest.pack_sequence_as(
      input_model._nested_outputs,  # pylint: disable=protected-access
      output_tensors,
  )

  return model_outputs, generator_outputs_list


def all_trainable_layers_are_registered(
    input_model: tf.keras.Model, layer_registry: lr.LayerRegistry
) -> bool:
  """Check if an input model's trainable layers are all registered.

  Args:
    input_model: The Keras model from which to obtain the layers from.
    layer_registry: A `LayerRegistry` instance containing functions that help
      compute gradient norms quickly. See
      `tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry` for
      more details.

  Returns:
    True if all the trainable layers in `input_model` are in `layer_registry`.
    False otherwise.
  """
  for layer in input_model.layers:
    if not layer_registry.is_elem(layer) and layer.trainable_variables:
      return False
  return True


def generate_model_outputs_using_core_keras_layers(
    input_model: tf.keras.Model,
    custom_layer_set: Optional[Set[type]] = None,  # pylint: disable=g-bare-generic
) -> type_aliases.PackedTensors:
  """Returns the model outputs generated by only core Keras layers.

  Args:
    input_model: A `tf.keras.Model` instance to obtain outputs from.
    custom_layer_set: An optional `set` of custom layers to expand. If `None`,
      then this is the set of all registered custom Keras layers.

  Returns:
    A `tf.Tensor` that is the result of `input_model(input_model.inputs)`
    using only Keras layers that are not in `custom_layer_set`.
  """
  # Set up helper variables and functions.
  custom_layer_set = (
      custom_layer_set or tf.keras.utils.get_custom_objects().values()
  )

  def _is_core(layer_instance):
    return type(layer_instance) not in custom_layer_set

  def generator_fn(layer_instance, args, kwargs):
    # Using `.call()` does not register the layer in the compute graph of
    # a forward pass.
    layer_outputs = (
        layer_instance(*args, **kwargs)
        if _is_core(layer_instance)
        else layer_instance.call(*args, **kwargs)
    )
    return layer_outputs, None

  # Return early if all the existing layers contain only core layers.
  if all(_is_core(layer) for layer in input_model.layers):
    return model_forward_pass(input_model, input_model.inputs)[0]

  # Do a forward pass to expand the outermost layers.
  candidate_outputs, _ = model_forward_pass(
      input_model, input_model.inputs, generator_fn
  )

  # The following recursion is inefficient because it recursively builds `n`
  # Keras model graphs, where `n` is the number of recursive calls. However,
  # it appears to be the only valid approach without accessing Keras's internal
  # functions (e.g., `keras.engine.functional._map_graph_network()`).
  cleaned_model = tf.keras.Model(
      inputs=input_model.inputs, outputs=candidate_outputs
  )
  return generate_model_outputs_using_core_keras_layers(
      cleaned_model, custom_layer_set
  )

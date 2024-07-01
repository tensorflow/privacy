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

import collections
from collections.abc import Callable, Sequence, Set
import dataclasses
from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases
from tensorflow_privacy.privacy.sparsity_preserving_noise import layer_registry as snlr
from tensorflow_privacy.privacy.sparsity_preserving_noise import type_aliases as sn_type_aliases


@dataclasses.dataclass(frozen=True)
class RegistryGeneratorFunctionOutput:
  layer_id: str
  layer_vars: Optional[Sequence[tf.Variable]]
  layer_sqr_norm_fn: Optional[type_aliases.SquareNormFunction]
  varname_to_count_contribution_fn: Optional[
      dict[str, sn_type_aliases.ContributionCountHistogramFn]
  ]
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
    sparse_noise_layer_registry: snlr.LayerRegistry,
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
    sparse_noise_layer_registry: A `LayerRegistry` instance containing functions
      that help compute contribution counts for sparse noise. See
      `tensorflow_privacy.privacy.sparsity_preserving_noise.layer_registry` for
      more details.
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
        varname_to_count_contribution_fn = None
        if sparse_noise_layer_registry and sparse_noise_layer_registry.is_elem(
            layer_instance
        ):
          count_contribution_registry_fn = sparse_noise_layer_registry.lookup(
              layer_instance
          )
          varname_to_count_contribution_fn = count_contribution_registry_fn(
              layer_instance, args, kwargs, num_microbatches
          )
        registry_fn = layer_registry.lookup(layer_instance)
        (layer_vars, layer_outputs, layer_sqr_norm_fn) = registry_fn(
            layer_instance, args, kwargs, tape, num_microbatches
        )
        return layer_outputs, RegistryGeneratorFunctionOutput(
            layer_id=str(id(layer_instance)),
            layer_vars=layer_vars,
            layer_sqr_norm_fn=layer_sqr_norm_fn,
            varname_to_count_contribution_fn=varname_to_count_contribution_fn,
            layer_trainable_weights=layer_instance.trainable_weights,
        )
      else:
        # Non-trainable layer.
        return layer_instance(*args, **kwargs), None

  return registry_generator_fn


def _infer_per_example_loss_fn(model: tf.keras.Model):
  """Infer the per-example loss from model config."""

  def _convert(loss_fn):
    loss_config = loss_fn.get_config()
    loss_config['reduction'] = tf.keras.losses.Reduction.NONE
    return loss_fn.from_config(loss_config)

  model_loss = model.loss
  if isinstance(model_loss, tf.keras.losses.Loss):
    return _convert(model_loss)
  elif isinstance(model_loss, dict):
    # Note that we cannot call the public method `.get_compile_config()` because
    # it calls a numpy function, which is not supported inside a `tf.function`
    # wrapped function.
    compile_config = model._compile_config.config  # pylint: disable=protected-access
    if compile_config is None:
      raise ValueError('Model must be compiled for loss function conversion')
    # Does a weighted mean of the configured losses. Note that we cannot build
    # from the config of the compiled loss because (i) it builds a
    # `keras.metrics.Mean` class, which generates non-unique `tf.Variable`s
    # during its construction, (ii) non-unique `tf.Variables` cannot be used
    # inside a `tf.function`, which is usually where this function is used.
    if 'loss_weights' not in compile_config:
      raise ValueError(
          'Models with multiple loss must have corresponding loss weights for'
          ' loss function conversion'
      )
    weights = compile_config['loss_weights']
    per_example_losses = {k: _convert(v) for k, v in model_loss.items()}
    num_losses = len(weights)

    def _per_example_loss_fn(y_true, y_pred, sample_weight=None):
      loss_values = []
      if model_loss.keys() - y_pred.keys():
        raise ValueError(
            'y_pred must contain the same keys and the model losses, but '
            'got %s and %s' % (y_pred.keys(), model_loss.keys())
        )
      if model_loss.keys() - y_true.keys():
        raise ValueError(
            'y_true must contain the same keys and the model losses, but '
            'got %s and %s' % (y_true.keys(), model_loss.keys())
        )
      if sample_weight is not None:
        if model_loss.keys() - sample_weight.keys():
          raise ValueError(
              'sample_weight must contain the same keys and the model losses,'
              ' but got %s and %s' % (y_true.keys(), model_loss.keys())
          )
      for k in y_true.keys():
        sgl_sample_weight = None if sample_weight is None else sample_weight[k]
        sgl_value = (
            weights[k]
            * per_example_losses[k](y_true[k], y_pred[k], sgl_sample_weight)
            / num_losses
        )
        loss_values.append(tf.reshape(sgl_value, shape=[-1]))
      return tf.math.add_n(loss_values)

    return _per_example_loss_fn
  else:
    raise ValueError(
        'Unsupported type for loss function conversion: {}'.format(
            type(model_loss)
        )
    )


def model_forward_backward_pass(
    tape: tf.GradientTape,
    input_model: tf.keras.Model,
    x_batch: type_aliases.InputTensors,
    y_batch: type_aliases.OutputTensors,
    registry_generator_fn: Optional[
        Callable[..., Tuple[tf.Tensor, RegistryGeneratorFunctionOutput]]
    ],
    weight_batch: Optional[tf.Tensor] = None,
    per_example_loss_fn: Optional[type_aliases.LossFn] = None,
    num_microbatches: Optional[type_aliases.BatchSize] = None,
    trainable_vars: Optional[Sequence[tf.Variable]] = None,
) -> tuple[
    dict[str, list[type_aliases.Tensor]], list[RegistryGeneratorFunctionOutput]
]:
  """Does a forward and backward pass of a model and returns useful intermediates."""
  # First loop computes the model outputs, summed loss, and generator outputs.
  with tape:
    model_outputs, generator_outputs_list = model_forward_pass(
        input_model, x_batch, generator_fn=registry_generator_fn
    )

    # Ignore the original loss function's reduction to get per-example loss.
    if per_example_loss_fn is None:
      per_example_loss_fn = _infer_per_example_loss_fn(input_model)

    losses = per_example_loss_fn(y_batch, model_outputs, weight_batch)
    if losses.shape is None:
      raise NotImplementedError(
          "The unreduced (or per-example) loss's shape cannot be `None`"
      )
    if len(losses.shape) != 1:
      raise NotImplementedError(
          'The unreduced (or per-example) loss needs to have a shape of length '
          'one, but received an unreduced loss of shape length %s'
          % len(losses.shape)
      )
    if num_microbatches is not None:
      losses = tf.reduce_mean(
          common_manip_utils.maybe_add_microbatch_axis(
              losses, num_microbatches
          ),
          axis=1,
      )
    summed_loss = tf.reduce_sum(losses)
  # Unwrap the generator outputs so that the next loop avoids duplicating
  # backprop ops.
  filtered_outputs = [t for t in generator_outputs_list if t is not None]

  if trainable_vars is not None:
    # Create a set using `ref()` for fast set membership check. tf.Variable
    # itself is not hashable.
    trainable_vars = set([v.ref() for v in trainable_vars])
  layer_vars = collections.defaultdict(list)
  for registry_fn_output in filtered_outputs:
    if trainable_vars is None or any(
        w.ref() in trainable_vars
        for w in registry_fn_output.layer_trainable_weights
    ):
      layer_vars[registry_fn_output.layer_id].append(
          registry_fn_output.layer_vars
      )

  layer_grad_vars = tape.gradient(
      summed_loss,
      layer_vars,
      unconnected_gradients=tf.UnconnectedGradients.ZERO,
  )
  if not layer_grad_vars:
    raise ValueError('The gradient list cannot be empty.')

  return layer_grad_vars, filtered_outputs


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

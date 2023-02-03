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

from absl import logging
import tensorflow as tf


def _has_internal_compute_graph(input_object):
  """Checks if input is a TF model and has a TF internal compute graph."""
  return (
      isinstance(input_object, tf.keras.Model)
      and hasattr(input_object, '_flatten_to_reference_inputs')
      and hasattr(input_object, '_tensor_usage_count')
      and hasattr(input_object, '_conform_to_reference_input')
      and hasattr(input_object, '_nodes_by_depth')
  )


def _get_internal_layers(input_layer):
  """Returns a list of layers that are nested within a given layer."""
  internal_layers = []
  if isinstance(input_layer, tf.keras.Model) and hasattr(input_layer, 'layers'):
    for layer in input_layer.layers:
      internal_layers.extend(_get_internal_layers(layer))
  else:
    internal_layers.append(input_layer)
  return internal_layers


def _trainable_layer_forward_pass(input_layer, inputs, tape, layer_registry):
  """Does a forward pass of a trainable layer and returns useful intermediates."""
  trainable_layer_hash = hash(input_layer.__class__)
  if trainable_layer_hash not in layer_registry:
    raise NotImplementedError(
        'Layer %s is not in the registry of known layers that can '
        'be used for efficient gradient clipping.'
        % input_layer.__class__.__name__
    )
  registry_fn = layer_registry[trainable_layer_hash]
  layer_vars, transform, layer_sqr_norm_fn = registry_fn(input_layer, inputs)
  if tape is not None:
    tape.watch(layer_vars)
  layer_outputs = transform(layer_vars) if transform else layer_vars
  return layer_outputs, layer_vars, layer_sqr_norm_fn


def model_forward_pass(
    input_model, inputs, tape, layer_registry, expand_custom_layers=False
):
  """Does a forward pass of a model and returns useful intermediates.

  NOTE: the graph traversal algorithm is an adaptation of the logic in the
    _run_internal_graph() method in the functional.Functional class. Hence,
    forward_norm_pass should only be invoked if the generated model
    instance is an instance of the functional.Functional class.

  Args:
    input_model: A `tf.keras.Model` to compute the quantities for.
    inputs: Arbitrary input to be fed into the input layer of the model. It is
      expected that `input_model(inputs)` returns a valid output.
    tape: `None` or a `tf.GradientTape` instance used to record certain
      operations. Assumes that this function is being called inside of `tape` if
      it is not `None`.
    layer_registry: `None`or a `dict` of layers that support "fast" gradient
      norm computations. The key is the class of the layer and the value is a
      function that returns a triple (norm_list, var_list, transform). For more
      details, see `layer_registry_factories.py`.
    expand_custom_layers: If `True`, then custom `tf.keras.layers.Layer` nodes
      in the compute graph of `input_model` are not registered in the compute
      graph that generates `outputs` in the output `tuple` (see below). Instead
      each custom layer `l` is only invoked with `l.call()` so only the
      underlying Keras operations are registered.

  Returns:
    A `tuple` `(outputs, base_var_list, sqr_norm_fn_list)`. `outputs` is the
    `tf.Tensor` that is generated as a result of a forward pass. `base_var_list`
    is an ordered list of `tf.Tensor` objects that are intended to be
    differentiated with respect to the summed loss of the model.
    `sqr_norm_fn_list` is either `None` or a `list` of functions that return
    the squared L2 gradient norms for a specific trainable layer.
    Specifically, the i-th function takes, as input, the output of
    `tape.gradient(summed_loss, base_var_list[i])` and returns the gradient
    norms for the layer corresponding to index `i`.
  """
  # TODO: Avoid or remove the references to protected methods of `input_model`.  # pylint: disable=g-bad-todo
  # Prepare the inputs and BFS variables.
  flattened_inputs = input_model._flatten_to_reference_inputs(inputs)  # pylint: disable=protected-access
  tensor_dict = {}
  tensor_usage_count = input_model._tensor_usage_count  # pylint: disable=protected-access
  for x, y in zip(input_model.inputs, flattened_inputs):
    y = input_model._conform_to_reference_input(y, ref_input=x)  # pylint: disable=protected-access
    x_id = str(id(x))
    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

  # Main computations.
  nodes_by_depth = input_model._nodes_by_depth  # pylint: disable=protected-access
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)
  base_var_list = []
  sqr_norm_fn_list = []
  node_outputs = None
  cust_obj_dict = dict.copy(tf.keras.utils.get_custom_objects())
  cust_hash_set = set([hash(v) for v in cust_obj_dict.values()])

  # Perform BFS feedforward computations.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      if node.is_input:
        continue  # inputs already exist
      if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
        continue  # node is not computable; try skipping
      args, kwargs = node.map_arguments(tensor_dict)
      if _has_internal_compute_graph(node.layer):
        # If this node has an internal computational graph, we can recurse.
        node_outputs, node_vars, node_sqr_norm_fn = model_forward_pass(
            node.layer, args, tape, layer_registry
        )
        base_var_list.extend(node_vars)
        sqr_norm_fn_list.extend(node_sqr_norm_fn)
      else:
        # Otherwise, we parse the node directly.
        layer_inputs = args
        layer_kwargs = kwargs
        node_layers = _get_internal_layers(node.layer)
        for layer in node_layers:
          if layer_registry is not None and layer.trainable_variables:
            raw_outputs, node_vars, layer_sqr_norm_fn = (
                _trainable_layer_forward_pass(
                    layer, layer_inputs, tape, layer_registry
                )
            )
            base_var_list.append(node_vars)
            sqr_norm_fn_list.append(layer_sqr_norm_fn)
          else:
            if hash(layer.__class__) in cust_hash_set and expand_custom_layers:
              raw_outputs = layer.call(*layer_inputs, **layer_kwargs)
            else:
              raw_outputs = layer(*layer_inputs, **layer_kwargs)

          layer_inputs = (raw_outputs,)
      # Update the current dictionary of inputs for the next node.
      node_outputs = layer_inputs
      for x_id, y in zip(node.flat_output_ids, tf.nest.flatten(node_outputs)):
        tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

  return node_outputs, base_var_list, sqr_norm_fn_list


def _all_trainable_layers_are_registered(input_model, layer_registry):
  """Check if an input model's trainable layers are all registered.

  Args:
    input_model: The Keras model from which to obtain the layers from.
    layer_registry: A dictionary of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a triple (output, sqr_grad_norms, vars), where
      output is the pre-activator tensor, sqr_grad_norms is the square of the
      norm of the layer's input, and vars is an ordered list of the trainable
      weights.

  Returns:
    True if all the trainable layers in `input_model` are in `layer_registry`.
    False otherwise.
  """
  for layer in input_model.layers:
    for sublayer in _get_internal_layers(layer):
      if (
          hash(sublayer.__class__) not in layer_registry
          and sublayer.trainable_variables
      ):
        return False
  return True


def add_aggregate_noise(
    input_model, x_batch, clipped_grads, l2_norm_clip, noise_multiplier
):
  """Adds noise to a collection of clipped gradients.

  The magnitude of the noise depends on the aggregation strategy of the
  input model's loss function.

  Args:
    input_model: The Keras model to obtain the layers from.
    x_batch: A collection of Tensors to be fed into the input layer of the
      model.
    clipped_grads: A list of tensors representing the clipped gradients.
    l2_norm_clip: Clipping norm (max L2 norm of each gradient).
    noise_multiplier: Ratio of the standard deviation to the clipping norm.

  Returns:
    A list of tensors containing the clipped gradients, but with the right
    amount of Gaussian noise added to them (depending on the reduction
    strategy of the loss function).
  """
  scale = l2_norm_clip
  if input_model.loss.reduction in [
      tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
      tf.keras.losses.Reduction.AUTO,
  ]:
    if input_model.loss.reduction == tf.keras.losses.Reduction.AUTO:
      logging.info('Assuming that the loss reduction is `SUM_OVER_BATCH_SIZE`.')
    if isinstance(x_batch, tf.Tensor):
      scale /= tf.cast(tf.shape(x_batch)[0], tf.float32)
    elif isinstance(x_batch, dict):
      batch_sizes = [
          tf.cast(tf.shape(v)[0], tf.float32) for v in x_batch.values()
      ]
      scale /= tf.math.reduce_min(batch_sizes)
    else:
      raise NotImplementedError(
          'Unknown container/class %s for input' % x_batch.__class__.__name__
      )

  def add_noise(g):
    return g + tf.random.normal(
        tf.shape(g), mean=0.0, stddev=noise_multiplier * scale
    )

  return tf.nest.map_structure(add_noise, clipped_grads)


def generate_model_outputs_using_core_keras_layers(input_model):
  """Returns the model outputs generated by only core Keras layers."""
  (purified_outputs, _, _) = model_forward_pass(
      input_model, input_model.inputs, None, None, expand_custom_layers=True
  )
  return purified_outputs

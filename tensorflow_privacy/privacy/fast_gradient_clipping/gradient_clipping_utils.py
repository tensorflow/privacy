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


def has_internal_compute_graph(input_object):
  """Checks if input is a TF model and has a TF internal compute graph."""
  return (
      isinstance(input_object, tf.keras.Model)
      and hasattr(input_object, '_flatten_to_reference_inputs')
      and hasattr(input_object, '_tensor_usage_count')
      and hasattr(input_object, '_conform_to_reference_input')
      and hasattr(input_object, '_nodes_by_depth')
  )


def forward_norm_pass(input_model, x_batch, tape, layer_registry):
  """Does a forward pass of a model and returns some useful intermediates.

  NOTE: the graph traversal algorithm is an adaptation of the logic in the
    _run_internal_graph() method in the functional.Functional class. Hence,
    forward_norm_pass should only be invoked if the generated model
    instance is an instance of the functional.Functional class.

  Args:
    input_model: A Keras functional model to compute the quantities for.
    x_batch: A collection of Tensors to be fed into the input layer of the
      model.
    tape: A tf.GradientTape() instance used to record certain operations.
      Assumes that this function is being called inside of this tape.
    layer_registry: A dictionary of layers that support "fast" gradient norm
      computations. The key is the class of the layer and the value is a
      function that returns a triple (norm_list, var_list, transform). For more
      details, see `layer_registry_factories.py`.

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
  flattened_inputs = input_model._flatten_to_reference_inputs(x_batch)  # pylint: disable=protected-access
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
        node_outputs, node_vars, node_sqr_norm_fn = forward_norm_pass(
            node.layer, args, tape, layer_registry
        )
        base_var_list.extend(node_vars)
        sqr_norm_fn_list.extend(node_sqr_norm_fn)
      else:
        # Either pass through or record some metadata.
        if not node.layer.trainable_variables:
          node_outputs = node.layer(*args, **kwargs)
        else:
          layer_hash = hash(node.layer.__class__)
          if layer_hash not in layer_registry:
            raise NotImplementedError(
                'Layer %s is not in the registry of known layers that can'
                'be used for efficient gradient clipping.'
                % node.layer.__class__.__name__
            )
          registry_fn = layer_registry[layer_hash]
          node_vars, transform, node_sqr_norm_fn = registry_fn(node.layer, args)
          tape.watch(node_vars)
          node_outputs = transform(node_vars) if transform else node_vars
          base_var_list.append(node_vars)
          sqr_norm_fn_list.append(node_sqr_norm_fn)
      # Update the current dictionary of inputs for the next node.
      for x_id, y in zip(node.flat_output_ids, tf.nest.flatten(node_outputs)):
        tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

  return node_outputs, base_var_list, sqr_norm_fn_list


def get_trainable_hidden_layers(input_model):
  """Obtains the trainable hidden layers of a Keras model.

  Args:
    input_model: The Keras model to obtain the layers from.

  Returns:
    A list of Keras layers where the tensorflow.keras.layers.Layer
    ancestor class MUST precede any existing tensorflow.keras.models.Model
    ancestor class.
  """
  hidden_layers = []
  for l in input_model.layers:
    for c in l.__class__.__mro__:
      if c == tf.keras.models.Model:
        hidden_layers += get_trainable_hidden_layers(l)
        break
      elif c == tf.keras.layers.InputLayer:
        break
      elif c == tf.keras.layers.Layer:
        if l.trainable_variables:
          hidden_layers.append(l)
        break
  return hidden_layers


def all_trainable_layers_are_registered(input_model, layer_registry):
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
  hidden_layers = get_trainable_hidden_layers(input_model)
  for l in hidden_layers:
    if hash(l.__class__) not in layer_registry:
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

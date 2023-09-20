# Copyright 2023, The TensorFlow Authors.
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
"""A collection of common utility functions for unit testing."""

from collections.abc import Callable, MutableSequence, Sequence
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow.compat.v2 as tf_compat
from tensorflow_privacy.privacy.fast_gradient_clipping import clip_grads
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


# ==============================================================================
# Helper functions
# ==============================================================================
def create_tpu_strategy():
  """Initializes a TPU environment."""
  # Done to avoid transferring data between CPUs and TPUs.
  tf_compat.config.set_soft_device_placement(False)
  resolver = tf_compat.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf_compat.config.experimental_connect_to_cluster(resolver)
  tf_compat.tpu.experimental.initialize_tpu_system(resolver)
  return tf_compat.distribute.TPUStrategy(resolver)


def assert_replica_values_are_close(test_case_obj, replica_context):
  """Checks if all replica context tensors are near each other."""
  base_tensor = replica_context.values[0]
  for t in replica_context.values[1:]:
    test_case_obj.assertAllClose(base_tensor, t)


def get_nd_test_batches(n: int):
  """Returns a list of input batches of dimension n."""
  # The first two batches have a single element, the last batch has 2 elements.
  x0 = tf.zeros([1, n], dtype=tf.float64)
  x1 = tf.constant([range(n)], dtype=tf.float64)
  x2 = tf.concat([x0, x1], axis=0)
  w0 = tf.constant([1], dtype=tf.float64)
  w1 = tf.constant([2], dtype=tf.float64)
  w2 = tf.constant([0.5, 0.5], dtype=tf.float64)
  return [x0, x1, x2], [w0, w1, w2]


def test_loss_fn(
    x: tf.Tensor, y: tf.Tensor, weights: Optional[tf.Tensor] = None
) -> tf.Tensor:
  # Define a loss function which is unlikely to be coincidently defined.
  if weights is None:
    weights = 1.0
  loss = 3.14 * tf.reduce_sum(
      tf.cast(weights, tf.float32) * tf.square(x - y), axis=1
  )
  return loss


def compute_true_gradient_norms(
    input_model: tf.keras.Model,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    weight_batch: Optional[tf.Tensor],
    per_example_loss_fn: Optional[type_aliases.LossFn],
    num_microbatches: Optional[int],
    trainable_vars: Optional[tf.Variable] = None,
) -> type_aliases.OutputTensors:
  """Computes the real gradient norms for an input `(model, x, y)`."""
  if per_example_loss_fn is None:
    loss_config = input_model.loss.get_config()
    loss_config['reduction'] = tf.keras.losses.Reduction.NONE
    per_example_loss_fn = input_model.loss.from_config(loss_config)
  with tf.GradientTape(persistent=True) as tape:
    y_pred = input_model(x_batch)
    loss = per_example_loss_fn(y_batch, y_pred, weight_batch)
    if num_microbatches is not None:
      loss = tf.reduce_mean(
          tf.reshape(
              loss,
              tf.concat([[num_microbatches, -1], tf.shape(loss)[1:]], axis=0),
          ),
          axis=1,
      )
    if isinstance(loss, tf.RaggedTensor):
      loss = loss.to_tensor()
  sqr_norms = []
  trainable_vars = trainable_vars or input_model.trainable_variables
  for var in trainable_vars:
    jacobian = tape.jacobian(loss, var, experimental_use_pfor=False)
    reduction_axes = tf.range(1, tf.rank(jacobian))
    sqr_norm = tf.reduce_sum(tf.square(jacobian), axis=reduction_axes)
    sqr_norms.append(sqr_norm)
  sqr_norm_tsr = tf.stack(sqr_norms, axis=1)
  return tf.sqrt(tf.reduce_sum(sqr_norm_tsr, axis=1))


def get_model_from_generator(
    model_generator: type_aliases.ModelGenerator,
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
    is_eager: bool,
) -> tf.keras.Model:
  """Creates a simple model from input specifications."""
  model = model_generator(layer_generator, input_dims, output_dims)
  model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
      loss=tf.keras.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
      ),
      run_eagerly=is_eager,
  )
  return model


def get_computed_and_true_norms_from_model(
    model: tf.keras.Model,
    per_example_loss_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    num_microbatches: Optional[int],
    x_batch: tf.Tensor,
    weight_batch: Optional[tf.Tensor] = None,
    rng_seed: int = 777,
    registry: layer_registry.LayerRegistry = None,
    partial: bool = False,
):
  """Generates relevant norms from an input model and other specs."""
  trainable_vars = None
  if partial:
    # Gets the first layer with variables.
    for l in model.layers:
      trainable_vars = l.trainable_variables
      if trainable_vars:
        break
  y_pred = model(x_batch)
  y_batch = tf.ones_like(y_pred)
  tf.keras.utils.set_random_seed(rng_seed)
  computed_norms = clip_grads.compute_gradient_norms(
      input_model=model,
      x_batch=x_batch,
      y_batch=y_batch,
      weight_batch=weight_batch,
      layer_registry=registry,
      per_example_loss_fn=per_example_loss_fn,
      num_microbatches=num_microbatches,
      trainable_vars=trainable_vars,
  )
  tf.keras.utils.set_random_seed(rng_seed)
  true_norms = compute_true_gradient_norms(
      input_model=model,
      x_batch=x_batch,
      y_batch=y_batch,
      weight_batch=weight_batch,
      per_example_loss_fn=per_example_loss_fn,
      num_microbatches=num_microbatches,
      trainable_vars=trainable_vars,
  )
  return computed_norms, true_norms


def get_computed_and_true_norms(
    model_generator: type_aliases.ModelGenerator,
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
    per_example_loss_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    num_microbatches: Optional[int],
    is_eager: bool,
    x_batch: tf.Tensor,
    weight_batch: Optional[tf.Tensor] = None,
    rng_seed: int = 777,
    registry: layer_registry.LayerRegistry = None,
    partial: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Obtains the true and computed gradient norms for a model and batch input.

  Helpful testing wrapper function used to avoid code duplication.

  Args:
    model_generator: A function which takes in three arguments:
      `layer_generator`, `idim`, and `odim`. Returns a `tf.keras.Model` that
      accepts input tensors of dimension `idim` and returns output tensors of
      dimension `odim`. Layers of the model are based on the `layer_generator`
      (see below for its description).
    layer_generator: A function which takes in two arguments: `idim` and `odim`.
      Returns a `tf.keras.layers.Layer` that accepts input tensors of dimension
      `idim` and returns output tensors of dimension `odim`.
    input_dims: The input dimension(s) of the test `tf.keras.Model` instance.
    output_dims: The output dimension(s) of the test `tf.keras.Model` instance.
    per_example_loss_fn: If not None, used as vectorized per example loss
      function.
    num_microbatches: The number of microbatches. None or an integer.
    is_eager: whether the model should be run eagerly.
    x_batch: inputs to be tested.
    weight_batch: optional weights passed to the loss.
    rng_seed: used as a seed for random initialization.
    registry: required for fast clipping.
    partial: Whether to compute the gradient norm with respect to a partial set
      of varibles. If True, only consider the variables in the first layer.

  Returns:
    A `tuple` `(computed_norm, true_norms)`. The first element contains the
    clipped gradient norms that are generated by
    `clip_grads.compute_gradient_norms()` under the setting given by the given
    model and layer generators. The second element contains the true clipped
    gradient norms under the aforementioned setting.
  """
  model = get_model_from_generator(
      model_generator=model_generator,
      layer_generator=layer_generator,
      input_dims=input_dims,
      output_dims=output_dims,
      is_eager=is_eager,
  )
  return get_computed_and_true_norms_from_model(
      model=model,
      per_example_loss_fn=per_example_loss_fn,
      num_microbatches=num_microbatches,
      x_batch=x_batch,
      weight_batch=weight_batch,
      rng_seed=rng_seed,
      registry=registry,
      partial=partial,
  )


def reshape_and_sum(tensor: tf.Tensor) -> tf.Tensor:
  """Reshapes and sums along non-batch dims to get the shape [None, 1]."""
  reshaped_2d = tf.reshape(tensor, [tf.shape(tensor)[0], -1])
  return tf.reduce_sum(reshaped_2d, axis=-1, keepdims=True)


# ==============================================================================
# Model generators.
# ==============================================================================
def make_one_layer_functional_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
) -> tf.keras.Model:
  """Creates a 1-layer sequential model."""
  inputs = tf.keras.Input(shape=input_dims)
  layer1 = layer_generator(input_dims, output_dims)
  temp1 = layer1(inputs)
  outputs = reshape_and_sum(temp1)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_two_layer_functional_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
) -> tf.keras.Model:
  """Creates a 2-layer sequential model."""
  inputs = tf.keras.Input(shape=input_dims)
  layer1 = layer_generator(input_dims, output_dims)
  temp1 = layer1(inputs)
  temp2 = tf.keras.layers.Dense(1)(temp1)
  outputs = reshape_and_sum(temp2)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_two_tower_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
) -> tf.keras.Model:
  """Creates a 2-layer 2-input functional model."""
  inputs1 = tf.keras.Input(shape=input_dims)
  layer1 = layer_generator(input_dims, output_dims)
  temp1 = layer1(inputs1)
  inputs2 = tf.keras.Input(shape=input_dims)
  layer2 = layer_generator(input_dims, output_dims)
  temp2 = layer2(inputs2)
  temp3 = tf.add(temp1, temp2)
  temp4 = tf.keras.layers.Dense(1)(temp3)
  outputs = reshape_and_sum(temp4)
  return tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)


def make_bow_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
) -> tf.keras.Model:
  """Creates a simple embedding bow model."""
  inputs = tf.keras.Input(shape=input_dims, dtype=tf.int32)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  if len(output_dims) != 1:
    raise ValueError('Expected `output_dims` to be of size 1.')
  emb_layer = layer_generator(input_dims, output_dims)
  feature_embs = emb_layer(inputs)
  # Embeddings add one extra dimension to its inputs, which combined with the
  # batch dimension at dimension 0, equals two additional dimensions compared
  # to the number of input dimensions. Here, we want to reduce over the output
  # space, but exclude the batch dimension.
  reduction_axes = range(1, len(input_dims) + 2)
  example_embs = tf.expand_dims(
      tf.reduce_sum(feature_embs, axis=reduction_axes), axis=-1
  )
  return tf.keras.Model(inputs=inputs, outputs=example_embs)


def make_dense_bow_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: Sequence[int],
    output_dims: Sequence[int],
) -> tf.keras.Model:
  """Creates an embedding bow model with a `Dense` layer."""
  inputs = tf.keras.Input(shape=input_dims, dtype=tf.int32)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  emb_layer = layer_generator(input_dims, output_dims)
  if len(output_dims) != 1:
    raise ValueError('Expected `output_dims` to be of size 1.')
  feature_embs = emb_layer(inputs)
  # Embeddings add one extra dimension to its inputs, which combined with the
  # batch dimension at dimension 0, equals two additional dimensions compared
  # to the number of input dimensions. Here, we want to reduce over the output
  # space, but exclude the batch dimension.
  reduction_axes = range(1, len(input_dims) + 2)
  example_embs = tf.expand_dims(
      tf.reduce_sum(feature_embs, axis=reduction_axes), axis=-1
  )
  outputs = tf.keras.layers.Dense(1)(example_embs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_weighted_bow_model(
    layer_generator: type_aliases.LayerGenerator,
    input_dims: MutableSequence[int],
    output_dims: MutableSequence[int],
) -> tf.keras.Model:
  """Creates a weighted embedding bow model."""
  # NOTE: This model only accepts dense input tensors.
  inputs = tf.keras.Input(shape=input_dims, dtype=tf.int32)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  emb_layer = layer_generator(input_dims, output_dims)
  if len(output_dims) != 1:
    raise ValueError('Expected `output_dims` to be of size 1.')
  feature_embs = emb_layer(inputs)
  # Use deterministic weights to avoid seeding issues on TPUs.
  feature_shape = input_dims + output_dims
  feature_weights = tf.expand_dims(
      tf.reshape(
          tf.range(np.product(feature_shape), dtype=tf.float32),
          feature_shape,
      ),
      axis=0,
  )
  weighted_embs = feature_embs * feature_weights
  # Embeddings add one extra dimension to its inputs, which combined with the
  # batch dimension at dimension 0, equals two additional dimensions compared
  # to the number of input dimensions. Here, we want to reduce over the output
  # space, but exclude the batch dimension.
  reduction_axes = range(1, len(input_dims) + 2)
  example_embs = tf.expand_dims(
      tf.reduce_sum(weighted_embs, axis=reduction_axes), axis=-1
  )
  outputs = tf.keras.layers.Dense(1)(example_embs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

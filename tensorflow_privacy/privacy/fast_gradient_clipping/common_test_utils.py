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

import itertools
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import clip_grads
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry


# ==============================================================================
# Type aliases
# ==============================================================================
LayerGenerator = Callable[[int, int], tf.keras.layers.Layer]

ModelGenerator = Callable[
    [LayerGenerator, Union[int, List[int]], int], tf.keras.Model
]


# ==============================================================================
# Helper functions
# ==============================================================================
def get_nd_test_tensors(n: int):
  """Returns a list of candidate tests for a given dimension n."""
  return [
      tf.zeros((n,), dtype=tf.float64),
      tf.convert_to_tensor(range(n), dtype_hint=tf.float64),
  ]


def get_nd_test_batches(n: int):
  """Returns a list of candidate input batches of dimension n."""
  result = []
  tensors = get_nd_test_tensors(n)
  for batch_size in range(1, len(tensors) + 1, 1):
    combinations = list(
        itertools.combinations(get_nd_test_tensors(n), batch_size)
    )
    result = result + [tf.stack(ts, axis=0) for ts in combinations]
  return result


def test_loss_fn(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  x = tf.reshape(x, (tf.shape(x)[0], -1))
  y = tf.reshape(y, (tf.shape(y)[0], -1))
  # Define a loss function which is unlikely to be coincidently defined.
  return 3.14 * tf.reduce_sum(tf.square(x - y), axis=1)


def compute_true_gradient_norms(
    input_model: tf.keras.Model,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    per_example_loss_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    num_microbatches: Optional[int],
    trainable_vars: Optional[tf.Variable] = None,
) -> layer_registry.OutputTensor:
  """Computes the real gradient norms for an input `(model, x, y)`."""
  if per_example_loss_fn is None:
    loss_config = input_model.loss.get_config()
    loss_config['reduction'] = tf.keras.losses.Reduction.NONE
    per_example_loss_fn = input_model.loss.from_config(loss_config)
  with tf.GradientTape(persistent=True) as tape:
    y_pred = input_model(x_batch)
    loss = per_example_loss_fn(y_batch, y_pred)
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
    reduction_axes = tf.range(1, len(jacobian.shape))
    sqr_norm = tf.reduce_sum(tf.square(jacobian), axis=reduction_axes)
    sqr_norms.append(sqr_norm)
  sqr_norm_tsr = tf.stack(sqr_norms, axis=1)
  return tf.sqrt(tf.reduce_sum(sqr_norm_tsr, axis=1))


def get_computed_and_true_norms(
    model_generator: ModelGenerator,
    layer_generator: LayerGenerator,
    input_dims: Union[int, List[int]],
    output_dim: int,
    per_example_loss_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    num_microbatches: Optional[int],
    is_eager: bool,
    x_input: tf.Tensor,
    rng_seed: int = 777,
    registry: layer_registry.LayerRegistry = None,
    partial: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
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
    output_dim: The output dimension of the test `tf.keras.Model` instance.
    per_example_loss_fn: If not None, used as vectorized per example loss
      function.
    num_microbatches: The number of microbatches. None or an integer.
    is_eager: A `bool` that is `True` if the model should be run eagerly.
    x_input: `tf.Tensor` inputs to be tested.
    rng_seed: An `int` used to initialize model weights.
    registry: A `layer_registry.LayerRegistry` instance.
    partial: Whether to compute the gradient norm with respect to a partial set
      of varibles. If True, only consider the variables in the first layer.

  Returns:
    A `tuple` `(computed_norm, true_norms)`. The first element contains the
    clipped gradient norms that are generated by
    `clip_grads.compute_gradient_norms()` under the setting given by the given
    model and layer generators. The second element contains the true clipped
    gradient norms under the aforementioned setting.
  """
  model = model_generator(layer_generator, input_dims, output_dim)
  model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
      loss=tf.keras.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
      ),
      run_eagerly=is_eager,
  )
  trainable_vars = None
  if partial:
    # Gets the first layer with variables.
    for l in model.layers:
      trainable_vars = l.trainable_variables
      if trainable_vars:
        break
  y_pred = model(x_input)
  y_batch = tf.ones_like(y_pred)
  tf.keras.utils.set_random_seed(rng_seed)
  computed_norms = clip_grads.compute_gradient_norms(
      model,
      x_input,
      y_batch,
      layer_registry=registry,
      per_example_loss_fn=per_example_loss_fn,
      num_microbatches=num_microbatches,
      trainable_vars=trainable_vars,
  )
  tf.keras.utils.set_random_seed(rng_seed)
  true_norms = compute_true_gradient_norms(
      model,
      x_input,
      y_batch,
      per_example_loss_fn,
      num_microbatches,
      trainable_vars=trainable_vars,
  )
  return (computed_norms, true_norms)


# ==============================================================================
# Model generators.
# ==============================================================================
def make_two_layer_sequential_model(layer_generator, input_dim, output_dim):
  """Creates a 2-layer sequential model."""
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(input_dim,)))
  model.add(layer_generator(input_dim, output_dim))
  model.add(tf.keras.layers.Dense(1))
  return model


def make_three_layer_sequential_model(layer_generator, input_dim, output_dim):
  """Creates a 3-layer sequential model."""
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(input_dim,)))
  layer1 = layer_generator(input_dim, output_dim)
  model.add(layer1)
  if isinstance(layer1, tf.keras.layers.Embedding):
    # Having multiple consecutive embedding layers does not make sense since
    # embedding layers only map integers to real-valued vectors.
    model.add(tf.keras.layers.Dense(output_dim))
  else:
    model.add(layer_generator(output_dim, output_dim))
  model.add(tf.keras.layers.Dense(1))
  return model


def make_two_layer_functional_model(layer_generator, input_dim, output_dim):
  """Creates a 2-layer 1-input functional model with a pre-output square op."""
  inputs = tf.keras.Input(shape=(input_dim,))
  layer1 = layer_generator(input_dim, output_dim)
  temp1 = layer1(inputs)
  temp2 = tf.square(temp1)
  outputs = tf.keras.layers.Dense(1)(temp2)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_two_tower_model(layer_generator, input_dim, output_dim):
  """Creates a 2-layer 2-input functional model."""
  inputs1 = tf.keras.Input(shape=(input_dim,))
  layer1 = layer_generator(input_dim, output_dim)
  temp1 = layer1(inputs1)
  inputs2 = tf.keras.Input(shape=(input_dim,))
  layer2 = layer_generator(input_dim, output_dim)
  temp2 = layer2(inputs2)
  temp3 = tf.add(temp1, temp2)
  outputs = tf.keras.layers.Dense(1)(temp3)
  return tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)


def make_bow_model(layer_generator, input_dims, output_dim):
  """Creates a 1-layer bow model."""
  del layer_generator
  inputs = tf.keras.Input(shape=input_dims)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  emb_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=output_dim)
  feature_embs = emb_layer(inputs)
  reduction_axes = tf.range(1, len(feature_embs.shape))
  example_embs = tf.expand_dims(
      tf.reduce_sum(feature_embs, axis=reduction_axes), axis=-1
  )
  return tf.keras.Model(inputs=inputs, outputs=example_embs)


def make_dense_bow_model(layer_generator, input_dims, output_dim):
  """Creates a 2-layer bow model."""
  del layer_generator
  inputs = tf.keras.Input(shape=input_dims)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  cardinality = 10
  emb_layer = tf.keras.layers.Embedding(
      input_dim=cardinality, output_dim=output_dim
  )
  feature_embs = emb_layer(inputs)
  reduction_axes = tf.range(1, len(feature_embs.shape))
  example_embs = tf.expand_dims(
      tf.reduce_sum(feature_embs, axis=reduction_axes), axis=-1
  )
  outputs = tf.keras.layers.Dense(1)(example_embs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_weighted_bow_model(layer_generator, input_dims, output_dim):
  """Creates a 1-layer weighted bow model."""
  # NOTE: This model only accepts dense input tensors.
  del layer_generator
  inputs = tf.keras.Input(shape=input_dims)
  # For the Embedding layer, input_dim is the vocabulary size. This should
  # be distinguished from the input_dim argument, which is the number of ids
  # in eache example.
  cardinality = 10
  emb_layer = tf.keras.layers.Embedding(
      input_dim=cardinality, output_dim=output_dim
  )
  feature_embs = emb_layer(inputs)
  feature_weights = tf.random.uniform(tf.shape(feature_embs))
  weighted_embs = feature_embs * feature_weights
  reduction_axes = tf.range(1, len(weighted_embs.shape))
  example_embs = tf.expand_dims(
      tf.reduce_sum(weighted_embs, axis=reduction_axes), axis=-1
  )
  outputs = tf.keras.layers.Dense(1)(example_embs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

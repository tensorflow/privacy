# Copyright 2021, The TensorFlow Authors.
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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.keras_models import dp_keras_model


def get_data():
  # Data is for hidden weights of [3, 1] and bias of 2.
  # With mean squared loss, we expect loss = 15^2 = 225, gradients of
  # weights = [90, 120], and gradient of bias = 30.
  data = np.array([[3, 4]])
  labels = np.matmul(data, [[3], [1]]) + 2
  return data, labels


class DPKerasModelTest(tf.test.TestCase, parameterized.TestCase):

  def testBaseline(self):
    """Tests that DPSequential works when DP-SGD has no effect."""
    train_data, train_labels = get_data()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=1.0e9,
        noise_multiplier=0.0,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros')
        ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=1)

    model_weights = model.get_weights()

    # Check parameters are as expected, taking into account the learning rate.
    self.assertAllClose(model_weights[0], [[0.90], [1.20]])
    self.assertAllClose(model_weights[1], [0.30])

  @parameterized.product(
      l2_norm_clip=(10.0, 40.0, 200.0),
      fast_clipping=(True, False),
  )
  def testClippingNorm(self, l2_norm_clip, fast_clipping):
    """Tests that clipping norm works."""
    train_data, train_labels = get_data()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros'
            ),
        ],
    )
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    expected_loss = loss(train_labels, model(train_data))
    results = model.fit(train_data, train_labels, epochs=1, batch_size=1)

    model_weights = model.get_weights()

    unclipped_gradient = np.sqrt(90**2 + 120**2 + 30**2)
    scale = min(1.0, l2_norm_clip / unclipped_gradient)
    expected_weights = np.array([[90], [120]]) * scale * learning_rate
    expected_bias = np.array([30]) * scale * learning_rate

    # Check parameters are as expected, taking into account the learning rate.
    self.assertAllClose(model_weights[0], expected_weights)
    self.assertAllClose(model_weights[1], expected_bias)

    # Check the value of the loss.
    actual_loss = results.history['loss'][0]
    self.assertAllClose(expected_loss, actual_loss)

  def _compute_expected_gradients(self, data, labels, w, l2_norm_clip,
                                  num_microbatches):
    batch_size = data.shape[0]
    if num_microbatches is None:
      num_microbatches = batch_size

    preds = np.matmul(data, np.expand_dims(w, axis=1))

    grads = 2 * data * (preds - labels)

    grads = np.reshape(grads,
                       [num_microbatches, batch_size // num_microbatches, -1])

    mb_grads = np.mean(grads, axis=1)
    mb_grad_norms = np.linalg.norm(mb_grads, axis=1)

    scale = np.minimum(l2_norm_clip / mb_grad_norms, 1.0)

    mb_grads = mb_grads * scale[:, np.newaxis]

    final_grads = np.mean(mb_grads, axis=0)
    return final_grads

  @parameterized.product(
      num_microbatches=(None, 1, 2, 4),
      fast_clipping=(False, True),
  )
  def testMicrobatches(self, num_microbatches, fast_clipping):
    l2_norm_clip = 1.0
    train_data = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    w = np.zeros((2))
    train_labels = np.array([[1.0], [3.0], [-2.0], [-4.0]])
    learning_rate = 1.0

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    # Simple linear model returns w * x.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                1, use_bias=False, kernel_initializer='zeros'
            ),
        ],
    )
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=4, shuffle=False)

    model_weights = np.squeeze(model.get_weights())

    effective_num_microbatches = (
        train_data.shape[0]
        if model._num_microbatches is None
        else num_microbatches
    )

    expected_grads = self._compute_expected_gradients(
        train_data, train_labels, w, l2_norm_clip, effective_num_microbatches
    )
    expected_weights = np.squeeze(-learning_rate * expected_grads)
    self.assertAllClose(model_weights, expected_weights)

  @parameterized.product(
      l2_norm_clip=(3.0, 5.0),
      noise_multiplier=(2.0, 4.0),
      num_microbatches=(None, 1, 2, 4),
      fast_clipping=(False, True),
  )
  def testNoiseMultiplier(
      self, l2_norm_clip, noise_multiplier, num_microbatches, fast_clipping
  ):
    # The idea behind this test is to start with a model whose parameters
    # are set to zero. We then run one step of a model that produces
    # an un-noised gradient of zero, and then compute the standard deviation
    # of the resulting weights to see if it matches the expected standard
    # deviation.

    # Data is one example of length 1000, set to zero, with label zero.
    train_data = np.zeros((4, 1000))
    train_labels = np.array([[0.0], [0.0], [0.0], [0.0]])

    learning_rate = 1.0

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros'
            ),
        ],
    )
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=4)

    effective_num_microbatches = num_microbatches or train_data.shape[0]

    model_weights = model.get_weights()
    measured_std = np.std(model_weights[0])
    expected_std = l2_norm_clip * noise_multiplier / effective_num_microbatches

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.
    self.assertNear(measured_std, expected_std, 0.1 * expected_std)

  # Simple check to make sure dimensions are correct when output has
  # dimension > 1.
  @parameterized.product(
      num_microbatches=(None, 1, 2),
      output_dimension=(2, 4),
      fast_clipping=(False, True),
  )
  def testMultiDimensionalOutput(
      self, num_microbatches, output_dimension, fast_clipping
  ):
    train_data = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    train_labels = np.array([[0], [1], [1], [0]])
    learning_rate = 1.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = dp_keras_model.DPSequential(
        l2_norm_clip=1.0e9,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                output_dimension, use_bias=False, kernel_initializer='zeros'
            ),
            tf.keras.layers.Dense(1),
        ],
    )
    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(train_data, train_labels, epochs=1, batch_size=4, shuffle=False)

  # Checks that calls to earlier API using `use_xla` as a positional argument
  # raise an exception.
  @parameterized.named_parameters(
      ('earlier API True', True),
      ('earlier API False', False),
  )
  def testEarlierAPIFails(self, use_xla):
    with self.assertRaises(ValueError):
      _ = dp_keras_model.DPSequential(
          1.0e9,
          0.0,
          use_xla,
          layers=[
              tf.keras.layers.InputLayer(input_shape=(2,)),
              tf.keras.layers.Dense(
                  2, use_bias=False, kernel_initializer='zeros'
              ),
              tf.keras.layers.Dense(1),
          ],
      )

  # Simple test to check that regularizer gradients are contributing to the
  # final gradient.
  @parameterized.named_parameters(
      ('fast_clipping', True),
      ('no_fast_clipping', False),
  )
  def testRegularizationGradient(self, fast_clipping):
    input_dim = 10
    batch_size = 2
    regularizer_multiplier = 0.025
    inputs = tf.keras.layers.Input((input_dim,))
    dense_lyr = tf.keras.layers.Dense(
        1,
        kernel_initializer='ones',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(regularizer_multiplier),
    )
    # Zero-out outputs to avoid contributions from the main loss function.
    outputs = tf.multiply(dense_lyr(inputs), 0.0)
    model = dp_keras_model.DPModel(
        inputs=inputs,
        outputs=outputs,
        l2_norm_clip=1e9,
        noise_multiplier=0.0,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
    )
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.SGD(1.0),
        run_eagerly=True,
    )
    x_batch = tf.reshape(
        tf.range(input_dim * batch_size, dtype=tf.float32),
        [batch_size, input_dim],
    )
    y_batch = tf.zeros([batch_size, 1])
    model.fit(x=x_batch, y=y_batch)

    self.assertAllClose(
        model.trainable_variables,
        tf.multiply(
            tf.ones_like(model.trainable_variables),
            1.0 - 2.0 * regularizer_multiplier,
        ),
    )

  # Simple test to check that custom input regularization does NOT contribute
  # to the gradient.
  @parameterized.named_parameters(
      ('fast_clipping', True),
      ('no_fast_clipping', False),
  )
  def testCustomRegularizationZeroGradient(self, fast_clipping):
    input_dim = 10
    batch_size = 2
    inputs = tf.keras.layers.Input((input_dim,))
    dense_lyr = tf.keras.layers.Dense(
        1,
        kernel_initializer='ones',
        use_bias=False,
    )
    # Zero-out outputs to avoid contributions from the main loss function.
    outputs = tf.multiply(dense_lyr(inputs), 0.0)
    model = dp_keras_model.DPModel(
        inputs=inputs,
        outputs=outputs,
        l2_norm_clip=1e9,
        noise_multiplier=0.0,
        layer_registry=layer_registry.make_default_layer_registry()
        if fast_clipping
        else None,
    )
    model.add_loss(tf.reduce_sum(inputs))
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.SGD(1.0),
        run_eagerly=True,
    )
    x_batch = tf.reshape(
        tf.range(input_dim * batch_size, dtype=tf.float32),
        [batch_size, input_dim],
    )
    y_batch = tf.zeros([batch_size, 1])
    model.fit(x=x_batch, y=y_batch)

    self.assertAllClose(
        model.trainable_variables, tf.ones_like(model.trainable_variables)
    )


if __name__ == '__main__':
  tf.test.main()

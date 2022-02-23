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

  @parameterized.named_parameters(
      ('l2_norm_clip 10.0', 10.0),
      ('l2_norm_clip 40.0', 40.0),
      ('l2_norm_clip 200.0', 200.0),
  )
  def testClippingNorm(self, l2_norm_clip):
    """Tests that clipping norm works."""
    train_data, train_labels = get_data()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros')
        ])
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=1)

    model_weights = model.get_weights()

    unclipped_gradient = np.sqrt(90**2 + 120**2 + 30**2)
    scale = min(1.0, l2_norm_clip / unclipped_gradient)
    expected_weights = np.array([[90], [120]]) * scale * learning_rate
    expected_bias = np.array([30]) * scale * learning_rate

    # Check parameters are as expected, taking into account the learning rate.
    self.assertAllClose(model_weights[0], expected_weights)
    self.assertAllClose(model_weights[1], expected_bias)

  def _compute_expected_gradients(self, data, labels, w, l2_norm_clip,
                                  num_microbatches):
    batch_size = data.shape[0]
    if num_microbatches is None:
      num_microbatches = batch_size

    preds = np.matmul(data, w)

    grads = 2 * data * (labels - preds)[:, np.newaxis]
    grads = np.reshape(grads,
                       [num_microbatches, batch_size // num_microbatches, -1])

    mb_grads = np.mean(grads, axis=1)
    mb_grad_norms = np.linalg.norm(mb_grads, axis=1)

    scale = np.minimum(l2_norm_clip / mb_grad_norms, 1.0)

    mb_grads = mb_grads * scale[:, np.newaxis]

    final_grads = np.mean(mb_grads, axis=0)
    return final_grads

  @parameterized.named_parameters(
      ('mb_test 0', 1.0, None),
      ('mb_test 1', 1.0, 1),
      ('mb_test 2', 1.0, 2),
      ('mb_test 4', 1.0, 4),
  )
  def testMicrobatches(self, l2_norm_clip, num_microbatches):
    train_data = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    w = np.zeros((2))
    train_labels = np.array([1.0, 3.0, -2.0, -4.0])
    learning_rate = 1.0

    expected_grads = self._compute_expected_gradients(train_data, train_labels,
                                                      w, l2_norm_clip,
                                                      num_microbatches)
    expected_weights = np.squeeze(learning_rate * expected_grads)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                1, use_bias=False, kernel_initializer='zeros')
        ])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=4, shuffle=False)

    model_weights = np.squeeze(model.get_weights())
    self.assertAllClose(model_weights, expected_weights)

  @parameterized.named_parameters(
      ('noise_multiplier 3 2 1', 3.0, 2.0, 1),
      ('noise_multiplier 5 4 1', 5.0, 4.0, 1),
      ('noise_multiplier 3 2 2', 3.0, 2.0, 2),
      ('noise_multiplier 5 4 2', 5.0, 4.0, 2),
      ('noise_multiplier 3 2 4', 3.0, 2.0, 4),
      ('noise_multiplier 5 4 4', 5.0, 4.0, 4),
  )
  def testNoiseMultiplier(self, l2_norm_clip, noise_multiplier,
                          num_microbatches):
    # The idea behind this test is to start with a model whose parameters
    # are set to zero. We then run one step of a model that produces
    # an un-noised gradient of zero, and then compute the standard deviation
    # of the resulting weights to see if it matches the expected standard
    # deviation.

    # Data is one example of length 1000, set to zero, with label zero.
    train_data = np.zeros((4, 1000))
    train_labels = np.array([0.0, 0.0, 0.0, 0.0])

    learning_rate = 1.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros')
        ])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=4)

    model_weights = model.get_weights()
    measured_std = np.std(model_weights[0])
    expected_std = l2_norm_clip * noise_multiplier / num_microbatches

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.
    self.assertNear(measured_std, expected_std, 0.1 * expected_std)

  # Simple check to make sure dimensions are correct when output has
  # dimension > 1.
  @parameterized.named_parameters(
      ('mb_test None 1', None, 1),
      ('mb_test 1 2', 1, 2),
      ('mb_test 2 2', 2, 2),
      ('mb_test 4 4', 4, 4),
  )
  def testMultiDimensionalOutput(self, num_microbatches, output_dimension):
    train_data = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    train_labels = np.array([0, 1, 1, 0])
    learning_rate = 1.0

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = dp_keras_model.DPSequential(
        l2_norm_clip=1.0e9,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                output_dimension, use_bias=False, kernel_initializer='zeros')
        ])
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
                  2, use_bias=False, kernel_initializer='zeros')
          ])

if __name__ == '__main__':
  tf.test.main()

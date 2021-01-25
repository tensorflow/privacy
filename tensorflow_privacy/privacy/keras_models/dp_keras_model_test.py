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
"""Tests for DP Keras Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

  @parameterized.named_parameters(
      ('noise_multiplier 3 2', 3.0, 2.0),
      ('noise_multiplier 5 4', 5.0, 4.0),
  )
  def testNoiseMultiplier(self, l2_norm_clip, noise_multiplier):
    # The idea behind this test is to start with a model whose parameters
    # are set to zero. We then run one step of a model that produces
    # an un-noised gradient of zero, and then compute the standard deviation
    # of the resulting weights to see if it matches the expected standard
    # deviation.

    # Data is one example of length 1000, set to zero, with label zero.
    train_data = np.zeros((1, 1000))
    train_labels = np.array([0.0])

    learning_rate = 1.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    # Simple linear model returns w * x + b.
    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(
                1, kernel_initializer='zeros', bias_initializer='zeros')
        ])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_data, train_labels, epochs=1, batch_size=1)

    model_weights = model.get_weights()
    measured_std = np.std(model_weights[0])
    expected_std = l2_norm_clip * noise_multiplier

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.
    self.assertNear(measured_std, expected_std, 0.1 * expected_std)


if __name__ == '__main__':
  tf.test.main()

# Copyright 2020, The TensorFlow Authors.
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
"""Tests for DP-enabled binary class heads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.estimators import binary_class_head
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


class DPBinaryClassHeadTest(tf.test.TestCase):
  """Tests for DP-enabled heads."""

  def _make_input_data(self, size):
    """Create raw input data."""
    feature_a = np.random.normal(4, 1, (size))
    feature_b = np.random.normal(5, 0.7, (size))
    feature_c = np.random.normal(6, 2, (size))
    noise = np.random.normal(0, 30, (size))
    features = {
        'feature_a': feature_a,
        'feature_b': feature_b,
        'feature_c': feature_c,
    }
    labels = np.array(
        np.power(feature_a, 3) + np.power(feature_b, 2) +
        np.power(feature_c, 1) + noise > 125).astype(int)
    return features, labels

  def _make_input_fn(self, features, labels, training, batch_size=16):

    def input_fn():
      """An input function for training or evaluating."""
      # Convert the inputs to a Dataset.
      dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

      # Shuffle if in training mode.
      if training:
        dataset = dataset.shuffle(1000)

      return dataset.batch(batch_size)

    return input_fn

  def _make_model_fn(self, head, optimizer, feature_columns):
    """Constructs and returns a model_fn using DPBinaryClassHead."""

    def model_fn(features, labels, mode, params, config=None):  # pylint: disable=unused-argument
      feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
      inputs = feature_layer(features)
      hidden_layer = tf.keras.layers.Dense(units=3, activation='relu')
      hidden_layer_values = hidden_layer(inputs)
      logits_layer = tf.keras.layers.Dense(
          units=head.logits_dimension, activation=None)
      logits = logits_layer(hidden_layer_values)
      return head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          trainable_variables=hidden_layer.trainable_weights +
          logits_layer.trainable_weights,
          optimizer=optimizer)

    return model_fn

  def testLoss(self):
    """Tests loss() returns per-example losses."""

    head = binary_class_head.DPBinaryClassHead()
    features = {'feature_a': np.full((4), 1.0)}
    labels = np.array([[1.0], [1.0], [1.0], [0.0]])
    logits = np.full((4, 1), 0.5)

    actual_loss = head.loss(labels, logits, features)
    expected_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

    self.assertEqual(actual_loss.shape, [4, 1])

    if tf.executing_eagerly():
      self.assertEqual(actual_loss.shape, [4, 1])
      self.assertAllClose(actual_loss, expected_loss)
      return

    self.assertAllClose(expected_loss, self.evaluate(actual_loss))

  def testCreateTPUEstimatorSpec(self):
    """Tests that an Estimator built with this head works."""

    train_features, train_labels = self._make_input_data(256)
    feature_columns = []
    for key in train_features:
      feature_columns.append(tf.feature_column.numeric_column(key=key))

    head = binary_class_head.DPBinaryClassHead()
    optimizer = DPKerasSGDOptimizer(
        learning_rate=0.5,
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=2)
    model_fn = self._make_model_fn(head, optimizer, feature_columns)
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    classifier.train(
        input_fn=self._make_input_fn(train_features, train_labels, True),
        steps=4)

    test_features, test_labels = self._make_input_data(64)
    classifier.evaluate(
        input_fn=self._make_input_fn(test_features, test_labels, False),
        steps=4)

    predict_features, predict_labels_ = self._make_input_data(64)
    classifier.predict(
        input_fn=self._make_input_fn(predict_features, predict_labels_, False))


if __name__ == '__main__':
  tf.test.main()

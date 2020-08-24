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

# Lint as: python3
"""An example for using keras_evaluation."""

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.keras_evaluation import MembershipInferenceCallback
from tensorflow_privacy.privacy.membership_inference_attack.keras_evaluation import run_attack_on_keras_model
from tensorflow_privacy.privacy.membership_inference_attack.utils import get_all_attack_results


GradientDescentOptimizer = tf.train.GradientDescentOptimizer

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_string('model_dir', None, 'Model directory.')


def cnn_model():
  """Define a CNN model."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8, strides=2, padding='same',
                             activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  return model


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  (train_data,
   train_labels), (test_data,
                   test_labels) = tf.keras.datasets.mnist.load_data()

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Get model, optimizer and specify loss.
  model = cnn_model()
  optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Get callback for membership inference attack.
  mia_callback = MembershipInferenceCallback(
      (train_data, train_labels),
      (test_data, test_labels),
      attack_types=[AttackType.THRESHOLD_ATTACK],
      tensorboard_dir=FLAGS.model_dir)

  # Train model with Keras
  model.fit(train_data, train_labels,
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            batch_size=FLAGS.batch_size,
            callbacks=[mia_callback],
            verbose=2)

  print('End of training attack:')
  attack_results = run_attack_on_keras_model(
      model,
      (train_data, train_labels),
      (test_data, test_labels),
      slicing_spec=SlicingSpec(entire_dataset=True, by_class=True),
      attack_types=[AttackType.THRESHOLD_ATTACK, AttackType.K_NEAREST_NEIGHBORS]
      )

  attack_properties, attack_values = get_all_attack_results(attack_results)
  print('\n'.join(['  %s: %.4f' % (', '.join(p), r) for p, r in
                   zip(attack_properties, attack_values)]))


if __name__ == '__main__':
  app.run(main)

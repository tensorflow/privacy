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
"""Implementation of a single-layer softmax classifier."""

from typing import List, Optional, Union, Tuple, Any

import tensorflow as tf
from tensorflow_privacy.privacy.logistic_regression import datasets


def single_layer_softmax_classifier(
    train_dataset: datasets.RegressionDataset,
    test_dataset: datasets.RegressionDataset,
    epochs: int,
    num_classes: int,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: Union[tf.keras.losses.Loss, str] = 'categorical_crossentropy',
    batch_size: int = 32,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> Tuple[Any, List[float]]:
  """Trains a single layer neural network classifier with softmax activation.

  Args:
    train_dataset: consists of num_train many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    test_dataset: consists of num_test many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    epochs: the number of epochs.
    num_classes: the number of classes.
    optimizer: a tf.keras optimizer.
    loss: a tf.keras loss function.
    batch_size: a positive integer.
    kernel_regularizer: a regularization function.

  Returns:
    List of test accuracies (one for each epoch) on test_dataset of model
    trained on train_dataset.
  """
  one_hot_train_labels = tf.one_hot(train_dataset.labels, num_classes)
  one_hot_test_labels = tf.one_hot(test_dataset.labels, num_classes)
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Dense(
          units=num_classes,
          activation='softmax',
          kernel_regularizer=kernel_regularizer))
  model.compile(optimizer, loss=loss, metrics=['accuracy'])
  history = model.fit(
      train_dataset.points,
      one_hot_train_labels,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(test_dataset.points, one_hot_test_labels),
      verbose=0)
  weights = model.layers[0].weights
  return weights, history.history['val_accuracy']

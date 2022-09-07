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
"""Functions for generating train and test data for logistic regression models.

Includes two types of datasets:
- Synthetic linearly separable labeled examples.
  Here, in the binary classification case, we generate training examples by
  first sampling a random weight vector w from a multivariate Gaussian
  distribution. Then, for each training example, we randomly sample a point x,
  also from a multivariate Gaussian distribution, and then set the label y equal
  to 1 if the inner product of w and x is positive, and equal to 0 otherwise. As
  such, the training data is linearly separable.
  More generally, in the case where there are num_classes many classes, we
  sample num_classes different w vectors. After sampling x, we will set its
  class label y to the class for which the corresponding w vector has the
  largest inner product with x.
- MNIST 10-class classification dataset.
"""

import dataclasses
from typing import Tuple, Optional

import numpy as np
from sklearn import preprocessing
import tensorflow as tf


@dataclasses.dataclass
class RegressionDataset:
  """Class for storing labeled examples for a regression dataset.

  Attributes:
    points: array of shape (num_examples, dimension) containing the points to be
      classified.
    labels: array of shape (num_examples,) containing the corresponding labels,
      each belonging to the set {0,1,...,num_classes-1}, where num_classes is
      the number of classes.
    weights: dimension by num_classes matrix containing coefficients of linear
      separator, where dimension is the dimension and num_classes is the number
      of classes.
  """
  points: np.ndarray
  labels: np.ndarray
  weights: Optional[np.ndarray]


def linearly_separable_labeled_examples(
    num_examples: int, weights: np.ndarray) -> RegressionDataset:
  """Generates num_examples labeled examples using separator given by weights.

  Args:
    num_examples: number of labeled examples to generate.
    weights: dimension by num_classes matrix containing coefficients of linear
      separator, where dimension is the dimension and num_classes is the number
      of classes.

  Returns:
    RegressionDataset consisting of points and labels. Each point has unit
      l2-norm.
  """
  dimension = weights.shape[0]
  # Generate points and normalize each to have unit l2-norm.
  points_non_normalized = np.random.normal(size=(num_examples, dimension))
  points = preprocessing.normalize(points_non_normalized)
  # Compute labels.
  labels = np.argmax(np.matmul(points, weights), axis=1)
  return RegressionDataset(points, labels, weights)


def synthetic_linearly_separable_data(
    num_train: int, num_test: int, dimension: int,
    num_classes: int) -> Tuple[RegressionDataset, RegressionDataset]:
  """Generates synthetic train and test data for logistic regression.

  Args:
    num_train: number of training data points.
    num_test: number of test data points.
    dimension: the dimension of the classification problem.
    num_classes: number of classes, assumed to be at least 2.

  Returns:
    train_dataset: num_train labeled examples, with unit l2-norm points.
    test_dataset: num_test labeled examples, with unit l2-norm points.
  """
  if num_classes < 2:
    raise ValueError(f'num_classes must be at least 2. It is {num_classes}.')

  # Generate weight vector.
  weights = np.random.normal(size=(dimension, num_classes))

  # Generate train labeled examples.
  train_dataset = linearly_separable_labeled_examples(num_train, weights)

  # Generate test labeled examples.
  test_dataset = linearly_separable_labeled_examples(num_test, weights)

  return (train_dataset, test_dataset)


def mnist_dataset() -> Tuple[RegressionDataset, RegressionDataset]:
  """Generates (normalized) train and test data for MNIST.

  Returns:
    train_dataset: MNIST labeled examples, with unit l2-norm points.
    test_dataset: MNIST labeled examples, with unit l2-norm points.
  """
  train_data, test_data = tf.keras.datasets.mnist.load_data()
  train_points_non_normalized, train_labels = train_data
  test_points_non_normalized, test_labels = test_data
  num_train = train_points_non_normalized.shape[0]
  num_test = test_points_non_normalized.shape[0]
  train_points_non_normalized = train_points_non_normalized.reshape(
      (num_train, -1))
  test_points_non_normalized = test_points_non_normalized.reshape(
      (num_test, -1))
  train_points = preprocessing.normalize(train_points_non_normalized)
  test_points = preprocessing.normalize(test_points_non_normalized)
  return (RegressionDataset(train_points, train_labels, None),
          RegressionDataset(test_points, test_labels, None))

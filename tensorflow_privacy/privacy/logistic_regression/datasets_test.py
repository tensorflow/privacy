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

import unittest

from absl.testing import parameterized
import numpy as np
from tensorflow_privacy.privacy.logistic_regression import datasets


class DatasetsTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, np.array([
          [1],
      ])), (2, np.array([
          [1],
      ])), (5, np.array([[-1, 1], [1, -1]])),
      (15, np.array([[-1, 1.5, 2.1], [1.3, -3.3, -7.1], [1.3, -3.3, -7.1]])))
  def test_linearly_separable_labeled_examples(self, num_examples, weights):
    dimension, num_classes = weights.shape
    dataset = datasets.linearly_separable_labeled_examples(
        num_examples, weights)
    self.assertEqual(dataset.points.shape, (num_examples, dimension))
    self.assertEqual(dataset.labels.shape, (num_examples,))
    product = np.matmul(dataset.points, weights)
    for i in range(num_examples):
      for j in range(num_classes):
        self.assertGreaterEqual(product[i, dataset.labels[i]], product[i, j])

  @parameterized.parameters((1, 1, 1, 2), (20, 5, 1, 2), (20, 5, 2, 2),
                            (1000, 10, 15, 10))
  def test_synthetic(self, num_train, num_test, dimension, num_classes):
    (train_dataset, test_dataset) = datasets.synthetic_linearly_separable_data(
        num_train, num_test, dimension, num_classes)
    self.assertEqual(train_dataset.points.shape, (num_train, dimension))
    self.assertEqual(train_dataset.labels.shape, (num_train,))
    self.assertEqual(test_dataset.points.shape, (num_test, dimension))
    self.assertEqual(test_dataset.labels.shape, (num_test,))
    # Check that each train and test point has unit l2-norm.
    for i in range(num_train):
      self.assertAlmostEqual(np.linalg.norm(train_dataset.points[i, :]), 1)
    for i in range(num_test):
      self.assertAlmostEqual(np.linalg.norm(test_dataset.points[i, :]), 1)
    # Check that each train and test label is in {0,...,num_classes-1}.
    self.assertTrue(np.all(np.isin(train_dataset.labels, range(num_classes))))
    self.assertTrue(np.all(np.isin(test_dataset.labels, range(num_classes))))

  def test_mnist_dataset(self):
    (train_dataset, test_dataset) = datasets.mnist_dataset()
    self.assertEqual(train_dataset.points.shape, (60000, 784))
    self.assertEqual(train_dataset.labels.shape, (60000,))
    self.assertEqual(test_dataset.points.shape, (10000, 784))
    self.assertEqual(test_dataset.labels.shape, (10000,))
    # Check that each train and test point has unit l2-norm.
    for i in range(train_dataset.points.shape[0]):
      self.assertAlmostEqual(np.linalg.norm(train_dataset.points[i, :]), 1)
    for i in range(test_dataset.points.shape[0]):
      self.assertAlmostEqual(np.linalg.norm(test_dataset.points[i, :]), 1)
    # Check that each train and test label is in {0,...,9}.
    self.assertTrue(np.all(np.isin(train_dataset.labels, range(10))))
    self.assertTrue(np.all(np.isin(test_dataset.labels, range(10))))


if __name__ == '__main__':
  unittest.main()

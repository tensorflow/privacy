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
from tensorflow_privacy.privacy.logistic_regression import datasets
from tensorflow_privacy.privacy.logistic_regression import single_layer_softmax


class SingleLayerSoftmaxTest(parameterized.TestCase):

  @parameterized.parameters(
      (5000, 500, 3, 40, 2, 0.05),
      (5000, 500, 4, 40, 2, 0.05),
      (10000, 1000, 3, 40, 4, 0.1),
      (10000, 1000, 4, 40, 4, 0.1),
  )
  def test_single_layer_softmax(self, num_train, num_test, dimension, epochs,
                                num_classes, tolerance):
    (train_dataset, test_dataset) = datasets.synthetic_linearly_separable_data(
        num_train, num_test, dimension, num_classes)
    _, accuracy = single_layer_softmax.single_layer_softmax_classifier(
        train_dataset, test_dataset, epochs, num_classes, 'sgd')
    self.assertAlmostEqual(accuracy[-1], 1, delta=tolerance)

if __name__ == '__main__':
  unittest.main()

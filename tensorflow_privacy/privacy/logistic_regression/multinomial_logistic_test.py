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
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.logistic_regression import datasets
from tensorflow_privacy.privacy.logistic_regression import multinomial_logistic


class MultinomialLogisticRegressionTest(parameterized.TestCase):

  @parameterized.parameters(
      (5000, 500, 3, 1, 1e-5, 40, 2, 0.05),
      (5000, 500, 4, 1, 1e-5, 40, 2, 0.05),
      (10000, 1000, 3, 1, 1e-5, 40, 4, 0.1),
      (10000, 1000, 4, 1, 1e-5, 40, 4, 0.1),
  )
  def test_logistic_objective_perturbation(self, num_train, num_test, dimension,
                                           epsilon, delta, epochs, num_classes,
                                           tolerance):
    (train_dataset, test_dataset) = datasets.synthetic_linearly_separable_data(
        num_train, num_test, dimension, num_classes)
    _, accuracy = multinomial_logistic.logistic_objective_perturbation(
        train_dataset, test_dataset, epsilon, delta, epochs, num_classes, 1)
    # Since the synthetic data is linearly separable, we expect the test
    # accuracy to come arbitrarily close to 1 as the number of training examples
    # grows.
    self.assertAlmostEqual(accuracy[-1], 1, delta=tolerance)

  @parameterized.parameters(
      (1, 1, 1e-5, 40, 1, 1e-2),
      (500, 0.1, 1e-5, 40, 50, 1e-2),
      (5000, 10, 1e-5, 40, 10, 1e-3),
  )
  def test_compute_dpsgd_noise_multiplier(self, num_train, epsilon, delta,
                                          epochs, batch_size, tolerance):
    noise_multiplier = multinomial_logistic.compute_dpsgd_noise_multiplier(
        num_train, epsilon, delta, epochs, batch_size, tolerance)
    epsilon_lower_bound = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        num_train, batch_size, noise_multiplier + tolerance, epochs, delta)[0]
    epsilon_upper_bound = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        num_train, batch_size, noise_multiplier - tolerance, epochs, delta)[0]
    self.assertLess(epsilon_lower_bound, epsilon)
    self.assertLess(epsilon, epsilon_upper_bound)

  @parameterized.parameters(
      (5000, 500, 3, 1, 1e-5, 40, 2, 0.05, 10, 10, 1),
      (5000, 500, 4, 1, 1e-5, 40, 2, 0.05, 10, 10, 1),
      (5000, 500, 3, 2, 1e-4, 40, 4, 0.1, 10, 10, 1),
      (5000, 500, 4, 2, 1e-4, 40, 4, 0.1, 10, 10, 1),
  )
  def test_logistic_dpsgd(self, num_train, num_test, dimension, epsilon, delta,
                          epochs, num_classes, tolerance, batch_size,
                          num_microbatches, clipping_norm):
    (train_dataset, test_dataset) = datasets.synthetic_linearly_separable_data(
        num_train, num_test, dimension, num_classes)
    _, accuracy = multinomial_logistic.logistic_dpsgd(
        train_dataset, test_dataset, epsilon, delta, epochs, num_classes,
        batch_size, num_microbatches, clipping_norm)
    # Since the synthetic data is linearly separable, we expect the test
    # accuracy to come arbitrarily close to 1 as the number of training examples
    # grows.
    self.assertAlmostEqual(accuracy[-1], 1, delta=tolerance)


if __name__ == '__main__':
  unittest.main()

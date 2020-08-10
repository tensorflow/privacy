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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.utils."""
from absl.testing import absltest
import numpy as np
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack_new as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec


def get_test_input(n_train, n_test):
  """Get example inputs for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      rng.randn(n_train, 5) + 0.2,
      rng.randn(n_test, 5) + 0.2, np.array([i % 5 for i in range(n_train)]),
      np.array([i % 5 for i in range(n_test)]))


class RunAttacksTest(absltest.TestCase):

  def test_run_attacks_size(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(),
        (AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION))

    self.assertLen(result.single_attack_results, 2)

  def test_run_attack_trained_sets_attack_type(self):
    result = mia.run_attack(
        get_test_input(100, 100), AttackType.LOGISTIC_REGRESSION)

    self.assertEqual(result.attack_type, AttackType.LOGISTIC_REGRESSION)

  def test_run_attack_threshold_sets_attack_type(self):
    result = mia.run_attack(
        get_test_input(100, 100), AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.attack_type, AttackType.THRESHOLD_ATTACK)

  def test_run_attack_threshold_calculates_correct_auc(self):
    result = mia.run_attack(
        AttackInputData(
            loss_train=np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6]),
            loss_test=np.array([1.1, 1.2, 1.3, 0.4, 1.5, 1.6])),
        AttackType.THRESHOLD_ATTACK)

    np.testing.assert_almost_equal(result.roc_curve.get_auc(), 0.83, decimal=2)

  def test_run_attack_by_slice(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(by_class=True),
        (AttackType.THRESHOLD_ATTACK,))

    self.assertLen(result.single_attack_results, 6)
    expected_slice = SingleSliceSpec(SlicingFeature.CLASS, 2)
    self.assertEqual(result.single_attack_results[3].slice_spec, expected_slice)


if __name__ == '__main__':
  absltest.main()

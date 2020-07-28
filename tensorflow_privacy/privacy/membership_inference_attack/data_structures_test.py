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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.data_structures."""
from absl.testing import absltest
import numpy as np
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import RocCurve


class AttackInputDataTest(absltest.TestCase):

  def test_get_loss(self):
    attack_input = AttackInputData(
        logits_train=np.array([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
        logits_test=np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 1])
    )

    np.testing.assert_equal(
        attack_input.get_loss_train().tolist(), [0.5, 0.2])
    np.testing.assert_equal(
        attack_input.get_loss_test().tolist(), [0.2, 0.5])

  def test_get_loss_explicitly_provided(self):
    attack_input = AttackInputData(
        loss_train=np.array([1.0, 3.0, 6.0]),
        loss_test=np.array([1.0, 4.0, 6.0]))

    np.testing.assert_equal(
        attack_input.get_loss_train().tolist(), [1.0, 3.0, 6.0])
    np.testing.assert_equal(
        attack_input.get_loss_test().tolist(), [1.0, 4.0, 6.0])

  def test_validator(self):
    self.assertRaises(ValueError,
                      AttackInputData(logits_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(logits_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData().validate)


class RocCurveTest(absltest.TestCase):

  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 0.5)

  def test_auc_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 1.0)

  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_attacker_advantage(), 0.0)

  def test_attacker_advantage_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 1.0)


if __name__ == '__main__':
  absltest.main()

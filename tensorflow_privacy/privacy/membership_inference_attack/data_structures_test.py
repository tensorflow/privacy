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
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature


class AttackInputDataTest(absltest.TestCase):

  def test_get_loss(self):
    attack_input = AttackInputData(
        logits_train=np.array([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
        logits_test=np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 1]))

    np.testing.assert_equal(attack_input.get_loss_train().tolist(), [0.5, 0.2])
    np.testing.assert_equal(attack_input.get_loss_test().tolist(), [0.2, 0.5])

  def test_get_loss_explicitly_provided(self):
    attack_input = AttackInputData(
        loss_train=np.array([1.0, 3.0, 6.0]),
        loss_test=np.array([1.0, 4.0, 6.0]))

    np.testing.assert_equal(attack_input.get_loss_train().tolist(),
                            [1.0, 3.0, 6.0])
    np.testing.assert_equal(attack_input.get_loss_test().tolist(),
                            [1.0, 4.0, 6.0])

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
    self.assertRaises(ValueError, AttackInputData().validate)


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


class SingleAttackResultTest(absltest.TestCase):

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.get_auc(), 0.5)

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.get_attacker_advantage(), 0.0)


class AttackResultsTest(absltest.TestCase):

  perfect_classifier_result: SingleAttackResult
  random_classifier_result: SingleAttackResult

  def __init__(self, *args, **kwargs):
    super(AttackResultsTest, self).__init__(*args, **kwargs)

    # ROC curve of a perfect classifier
    self.perfect_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, True),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 1.0, 1.0]),
            fpr=np.array([1.0, 1.0, 0.0]),
            thresholds=np.array([0, 1, 2])))

    # ROC curve of a random classifier
    self.random_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2])))

  def test_get_result_with_max_auc_first(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(results.get_result_with_max_auc(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_auc_second(self):
    results = AttackResults(
        [self.random_classifier_result, self.perfect_classifier_result])
    self.assertEqual(results.get_result_with_max_auc(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_attacker_advantage_first(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(results.get_result_with_max_attacker_advantage(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_attacker_advantage_second(self):
    results = AttackResults(
        [self.random_classifier_result, self.perfect_classifier_result])
    self.assertEqual(results.get_result_with_max_attacker_advantage(),
                     self.perfect_classifier_result)

  def test_summary_by_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(
        results.summary(by_slices=True), 'Highest AUC on slice '
                                         'SingleSliceSpec' +
        '(SlicingFeature.CORRECTLY_CLASSIFIED=True) achieved by ' +
        'AttackType.THRESHOLD_ATTACK with an AUC of 1.0\n' +
        'Highest advantage on ' +
        'slice SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED=True) ' +
        'achieved by AttackType.THRESHOLD_ATTACK with an advantage of 1.0\n' +
        'Highest AUC on slice SingleSliceSpec(Entire dataset) achieved ' +
        'by AttackType.THRESHOLD_ATTACK with an AUC of 0.5\n' +
        'Highest advantage on slice SingleSliceSpec(Entire dataset) achieved ' +
        'by AttackType.THRESHOLD_ATTACK with an advantage of 0.0')

  def test_summary_without_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(
        results.summary(by_slices=False),
        'Highest AUC on slice SingleSliceSpec(Entire dataset) achieved ' +
        'by AttackType.THRESHOLD_ATTACK with an AUC of 0.5\n' +
        'Highest advantage on slice SingleSliceSpec(Entire dataset) achieved ' +
        'by AttackType.THRESHOLD_ATTACK with an advantage of 0.0')


if __name__ == '__main__':
  absltest.main()

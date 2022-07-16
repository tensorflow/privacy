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

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import _log_value
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import DataSize
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import LossFunction
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleMembershipProbabilityResult
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingFeature


class SingleSliceSpecTest(parameterized.TestCase):

  def testStrEntireDataset(self):
    self.assertEqual(str(SingleSliceSpec()), 'Entire dataset')

  @parameterized.parameters(
      (SlicingFeature.CLASS, 2, 'CLASS=2'),
      (SlicingFeature.PERCENTILE, (10, 20), 'Loss percentiles: 10-20'),
      (SlicingFeature.CORRECTLY_CLASSIFIED, True, 'CORRECTLY_CLASSIFIED=True'),
  )
  def testStr(self, feature, value, expected_str):
    self.assertEqual(str(SingleSliceSpec(feature, value)), expected_str)


class AttackInputDataTest(parameterized.TestCase):

  def test_get_xe_loss_from_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[-0.3, 1.5, 0.2], [2, 3, 0.5]]),
        logits_test=np.array([[2, 0.3, 0.2], [0.3, -0.5, 0.2]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 2]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(), [0.36313551, 1.37153903], atol=1e-7)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), [0.29860897, 0.95618669], atol=1e-7)

  def test_get_xe_loss_from_probs(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.1, 0.1, 0.8], [0.8, 0.2, 0]]),
        probs_test=np.array([[0, 0.0001, 0.9999], [0.07, 0.18, 0.75]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 2]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(), [2.30258509, 0.2231436], atol=1e-7)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), [18.42068074, 0.28768207], atol=1e-7)

  def test_get_binary_xe_loss_from_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([-10, -5, 0., 5, 10]),
        logits_test=np.array([-10, -5, 0., 5, 10]),
        labels_train=np.zeros((5,)),
        labels_test=np.ones((5,)),
        loss_function_using_logits=True)
    expected_loss0 = np.array([0.000045398, 0.006715348, 0.6931471825, 5, 10])
    np.testing.assert_allclose(
        attack_input.get_loss_train(), expected_loss0, rtol=1e-2)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), expected_loss0[::-1], rtol=1e-2)

  def test_get_binary_xe_loss_from_probs(self):
    attack_input = AttackInputData(
        probs_train=np.array([0.2, 0.7, 0.1, 0.99, 0.002, 0.008]),
        probs_test=np.array([0.2, 0.7, 0.1, 0.99, 0.002, 0.008]),
        labels_train=np.zeros((6,)),
        labels_test=np.ones((6,)),
        loss_function_using_logits=False)

    expected_loss0 = np.array([
        0.2231435513, 1.2039728043, 0.1053605157, 4.6051701860, 0.0020020027,
        0.0080321717
    ])
    expected_loss1 = np.array([
        1.6094379124, 0.3566749439, 2.3025850930, 0.0100503359, 6.2146080984,
        4.8283137373
    ])
    np.testing.assert_allclose(
        attack_input.get_loss_train(), expected_loss0, atol=1e-7)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), expected_loss1, atol=1e-7)

  @parameterized.named_parameters(
      ('use_logits', True, np.array([1, 0.]), np.array([0, 4.])),
      ('use_default', None, np.array([1, 0.]), np.array([0, 4.])),
      ('use_probs', False, np.array([0, 1.]), np.array([1, 1.])),
  )
  def test_get_squared_loss(self, loss_function_using_logits, expected_train,
                            expected_test):
    attack_input = AttackInputData(
        logits_train=np.array([0, 0.]),
        logits_test=np.array([0, 0.]),
        probs_train=np.array([1, 1.]),
        probs_test=np.array([1, 1.]),
        labels_train=np.array([1, 0.]),
        labels_test=np.array([0, 2.]),
        loss_function=LossFunction.SQUARED,
        loss_function_using_logits=loss_function_using_logits,
    )
    np.testing.assert_allclose(attack_input.get_loss_train(), expected_train)
    np.testing.assert_allclose(attack_input.get_loss_test(), expected_test)

  @parameterized.named_parameters(
      ('use_logits', True, np.array([125.]), np.array([121.])),
      ('use_default', None, np.array([125.]), np.array([121.])),
      ('use_probs', False, np.array([458.]), np.array([454.])),
  )
  def test_get_customized_loss(self, loss_function_using_logits, expected_train,
                               expected_test):

    def fake_loss(x, y):
      return 2 * x + y

    attack_input = AttackInputData(
        logits_train=np.array([
            123.,
        ]),
        logits_test=np.array([
            123.,
        ]),
        probs_train=np.array([
            456.,
        ]),
        probs_test=np.array([
            456.,
        ]),
        labels_train=np.array([1.]),
        labels_test=np.array([-1.]),
        loss_function=fake_loss,
        loss_function_using_logits=loss_function_using_logits,
    )
    np.testing.assert_allclose(attack_input.get_loss_train(), expected_train)
    np.testing.assert_allclose(attack_input.get_loss_test(), expected_test)

  @parameterized.named_parameters(
      ('both', np.array([0, 0.]), np.array([1, 1.]), np.array([1, 0.])),
      ('only_logits', np.array([0, 0.]), None, np.array([1, 0.])),
      ('only_probs', None, np.array([1, 1.]), np.array([0, 1.])),
  )
  def test_default_loss_function_using_logits(self, logits, probs, expected):
    """Tests for `loss_function_using_logits = None`. Should prefer logits."""
    attack_input = AttackInputData(
        logits_train=logits,
        logits_test=logits,
        probs_train=probs,
        probs_test=probs,
        labels_train=np.array([1, 0.]),
        labels_test=np.array([1, 0.]),
        loss_function=LossFunction.SQUARED,
    )
    np.testing.assert_allclose(attack_input.get_loss_train(), expected)
    np.testing.assert_allclose(attack_input.get_loss_test(), expected)

  @parameterized.parameters(
      (None, np.array([1.]), True),
      (np.array([1.]), None, False),
  )
  def test_loss_wrong_input(self, logits, probs, loss_function_using_logits):
    attack_input = AttackInputData(
        logits_train=logits,
        logits_test=logits,
        probs_train=probs,
        probs_test=probs,
        labels_train=np.array([
            1.,
        ]),
        labels_test=np.array([0.]),
        loss_function_using_logits=loss_function_using_logits,
    )
    self.assertRaises(ValueError, attack_input.get_loss_train)
    self.assertRaises(ValueError, attack_input.get_loss_test)

  def test_get_loss_explicitly_provided(self):
    attack_input = AttackInputData(
        loss_train=np.array([1.0, 3.0, 6.0]),
        loss_test=np.array([1.0, 4.0, 6.0]))

    np.testing.assert_equal(attack_input.get_loss_train().tolist(),
                            [1.0, 3.0, 6.0])
    np.testing.assert_equal(attack_input.get_loss_test().tolist(),
                            [1.0, 4.0, 6.0])

  def test_get_probs_sizes(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.1, 0.1, 0.8], [0.8, 0.2, 0]]),
        probs_test=np.array([[0, 0.0001, 0.9999]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0]))

    np.testing.assert_equal(attack_input.get_train_size(), 2)
    np.testing.assert_equal(attack_input.get_test_size(), 1)

  def test_get_entropy(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        logits_test=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        labels_train=np.array([0, 2]),
        labels_test=np.array([0, 2]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(), [0, 0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(),
                            [2 * _log_value(0), 0])

    attack_input = AttackInputData(
        logits_train=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        logits_test=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(), [0, 0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(), [0, 0])

  def test_get_entropy_explicitly_provided(self):
    attack_input = AttackInputData(
        entropy_train=np.array([0.0, 2.0, 1.0]),
        entropy_test=np.array([0.5, 3.0, 5.0]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(),
                            [0.0, 2.0, 1.0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(),
                            [0.5, 3.0, 5.0])

  def test_validator(self):
    self.assertRaises(ValueError,
                      AttackInputData(logits_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(probs_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(entropy_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(logits_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(probs_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(entropy_test=np.array([])).validate)
    self.assertRaises(ValueError, AttackInputData().validate)
    # Tests that having both logits and probs are not allowed.
    self.assertRaises(
        ValueError,
        AttackInputData(
            logits_train=np.array([]),
            logits_test=np.array([]),
            probs_train=np.array([]),
            probs_test=np.array([])).validate)

  def test_multilabel_validator(self):
    # Tests for multilabel data.
    with self.assertRaises(
        ValueError,
        msg='Validation passes incorrectly when `logits_test` is not 1D/2D.'):
      AttackInputData(
          logits_train=np.array([[-1.0, -2.0], [0.01, 1.5], [0.5, -3]]),
          logits_test=np.array([[[0.01, 1.5], [0.5, -3]],
                                [[0.01, 1.5], [0.5, -3]]]),
          labels_train=np.array([[0, 0], [0, 1], [1, 0]]),
          labels_test=np.array([[1, 1], [1, 0]]),
      ).validate()
    self.assertTrue(
        AttackInputData(
            logits_train=np.array([[-1.0, -2.0], [0.01, 1.5], [0.5, -3]]),
            logits_test=np.array([[0.01, 1.5], [0.5, -3]]),
            labels_train=np.array([[0, 0], [0, 1], [1, 1]]),
            labels_test=np.array([[1, 1], [1, 0]]),
        ).is_multilabel_data(),
        msg='Multilabel data check fails even though conditions are met.')

  def test_multihot_labels_check_on_null_array_returns_false(self):
    self.assertFalse(
        AttackInputData(
            logits_train=np.array([[-1.0, -2.0], [0.01, 1.5], [0.5, -3]]),
            logits_test=np.array([[0.01, 1.5], [0.5, -3]]),
            labels_train=np.array([[0, 0], [0, 1], [1, 1]]),
            labels_test=np.array([[1, 1], [1, 0]]),
        ).is_multihot_labels(None, 'null_array'),
        msg='Multilabel test on a null array should return False.')
    self.assertFalse(
        AttackInputData(
            logits_train=np.array([[-1.0, -2.0], [0.01, 1.5], [0.5, -3]]),
            logits_test=np.array([[0.01, 1.5], [0.5, -3]]),
            labels_train=np.array([[0, 0], [0, 1], [1, 1]]),
            labels_test=np.array([[1, 1], [1, 0]]),
        ).is_multihot_labels(np.array([1.0, 2.0, 3.0]), '1d_array'),
        msg='Multilabel test on a 1-D array should return False.')

  def test_multilabel_get_bce_loss_from_probs(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
        probs_test=np.array([[0.8, 0.7, 0.9]]),
        labels_train=np.array([[0, 1, 1], [1, 1, 0]]),
        labels_test=np.array([[1, 1, 0]]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(), [[0.22314343, 1.20397247, 0.3566748],
                                        [0.22314343, 0.51082546, 2.30258409]],
        atol=1e-6)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), [[0.22314354, 0.35667493, 2.30258499]],
        atol=1e-6)

  def test_multilabel_get_bce_loss_from_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[-1.0, -2.0], [0.01, 1.5], [0.5, -3]]),
        logits_test=np.array([[0.01, 1.5], [0.5, -3]]),
        labels_train=np.array([[0, 0], [0, 1], [1, 1]]),
        labels_test=np.array([[1, 1], [1, 0]]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(),
        [[0.31326167, 0.126928], [0.69815966, 0.20141327],
         [0.47407697, 3.04858714]],
        atol=1e-6)
    np.testing.assert_allclose(
        attack_input.get_loss_test(),
        [[0.68815966, 0.20141327], [0.47407697, 0.04858734]],
        atol=1e-6)

  def test_multilabel_get_loss_explicitly_provided(self):
    attack_input = AttackInputData(
        loss_train=np.array([[1.0, 3.0, 6.0], [6.0, 8.0, 9.0]]),
        loss_test=np.array([[1.0, 4.0, 6.0], [1.0, 2.0, 3.0]]))

    np.testing.assert_equal(attack_input.get_loss_train().tolist(),
                            np.array([[1.0, 3.0, 6.0], [6.0, 8.0, 9.0]]))
    np.testing.assert_equal(attack_input.get_loss_test().tolist(),
                            np.array([[1.0, 4.0, 6.0], [1.0, 2.0, 3.0]]))

  def test_validate_with_force_multilabel_false(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
        probs_test=np.array([[0.8, 0.7, 0.9]]),
        labels_train=np.array([[0, 0, 1], [0, 1, 0]]),
        labels_test=np.array([[1, 0, 0]]))
    self.assertRaisesRegex(ValueError,
                           r'should be a one dimensional numpy array.',
                           attack_input.validate)

  def test_validate_with_force_multilabel_true(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
        probs_test=np.array([[0.8, 0.7, 0.9]]),
        labels_train=np.array([[0, 0, 1], [0, 1, 0]]),
        labels_test=np.array([[1, 0, 0]]),
        force_multilabel_data=True)
    try:
      attack_input.validate()
    except ValueError:
      # For a 'ValueError' exception the test should record a failure. All
      # other exceptions are errors.
      self.fail('ValueError not raised by validate().')

  def test_multilabel_data_true_with_force_multilabel_true(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
        probs_test=np.array([[0.8, 0.7, 0.9]]),
        labels_train=np.array([[0, 0, 1], [0, 1, 0]]),
        labels_test=np.array([[1, 0, 0]]),
        force_multilabel_data=True)
    self.assertTrue(
        attack_input.multilabel_data,
        '"force_multilabel_data" is True but "multilabel_data" is False.')


class RocCurveTest(parameterized.TestCase):

  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_auc(), 0.5)

  def test_auc_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_auc(), 1.0)

  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_attacker_advantage(), 0.0)

  def test_attacker_advantage_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_auc(), 1.0)

  def test_ppv_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_ppv(), 0.5)

  def test_ppv_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    self.assertEqual(roc.get_ppv(), 1.0)

  # Parameters to test: test-train ratio, expected PPV.
  @parameterized.named_parameters(
      ('test_train_ratio_small', 0.001, 1.0),
      ('test_train_ratio_large', 1000.0, 0.0),
  )
  @mock.patch(
      'tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures._ABSOLUTE_TOLERANCE',
      1e-4)
  def test_ppv_perfect_classifier_when_tpr_fpr_small(self, test_train_ratio,
                                                     expected_ppv):
    roc = RocCurve(
        tpr=np.array([0.00001, 0.0001, 0.002]),
        fpr=np.array([0.00002, 0.0002, 0.002]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=test_train_ratio)

    np.testing.assert_allclose(roc.get_ppv(), expected_ppv, atol=1e-3)

  @mock.patch(
      'tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures._ABSOLUTE_TOLERANCE',
      1e-4)
  def test_ppv_random_classifier_when_tpr_fpr_small_and_test_train_is_1(self):
    roc = RocCurve(
        tpr=np.array([0.00001, 0.0001, 0.002]),
        fpr=np.array([0.00002, 0.0002, 0.002]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    np.testing.assert_allclose(roc.get_ppv(), 0.5, atol=1e-3)


class SingleAttackResultTest(absltest.TestCase):

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        data_size=DataSize(ntrain=1, ntest=1))

    self.assertEqual(result.get_auc(), 0.5)

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        data_size=DataSize(ntrain=1, ntest=1))

    self.assertEqual(result.get_attacker_advantage(), 0.0)

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_ppv_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]),
        test_train_ratio=1.0)

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        data_size=DataSize(ntrain=1, ntest=1))

    self.assertEqual(result.get_ppv(), 0.5)


class SingleMembershipProbabilityResultTest(absltest.TestCase):

  # Only a basic test to check the attack by setting a threshold on
  # membership probability.
  def test_attack_with_varied_thresholds(self):

    result = SingleMembershipProbabilityResult(
        slice_spec=SingleSliceSpec(None),
        train_membership_probs=np.array([0.91, 1, 0.92, 0.82, 0.75]),
        test_membership_probs=np.array([0.81, 0.7, 0.75, 0.25, 0.3]))

    self.assertEqual(
        result.attack_with_varied_thresholds(
            threshold_list=np.array([0.8, 0.7]))[1].tolist(), [0.8, 0.625])
    self.assertEqual(
        result.attack_with_varied_thresholds(
            threshold_list=np.array([0.8, 0.7]))[2].tolist(), [0.8, 1])


class AttackResultsCollectionTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.some_attack_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2]),
            test_train_ratio=1.0),
        data_size=DataSize(ntrain=1, ntest=1))

    self.results_epoch_10 = AttackResults(
        single_attack_results=[self.some_attack_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.4,
            accuracy_test=0.3,
            epoch_num=10,
            model_variant_label='default'))

    self.results_epoch_15 = AttackResults(
        single_attack_results=[self.some_attack_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.5,
            accuracy_test=0.4,
            epoch_num=15,
            model_variant_label='default'))

    self.attack_results_no_metadata = AttackResults(
        single_attack_results=[self.some_attack_result])

    self.collection_with_metadata = AttackResultsCollection(
        [self.results_epoch_10, self.results_epoch_15])

    self.collection_no_metadata = AttackResultsCollection(
        [self.attack_results_no_metadata, self.attack_results_no_metadata])

  def test_save_with_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
      self.collection_with_metadata.save(tmpdirname)
      loaded_collection = AttackResultsCollection.load(tmpdirname)

    self.assertEqual(
        repr(self.collection_with_metadata), repr(loaded_collection))
    self.assertLen(loaded_collection.attack_results_list, 2)

  def test_save_no_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
      self.collection_no_metadata.save(tmpdirname)
      loaded_collection = AttackResultsCollection.load(tmpdirname)

    self.assertEqual(repr(self.collection_no_metadata), repr(loaded_collection))
    self.assertLen(loaded_collection.attack_results_list, 2)


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
            thresholds=np.array([0, 1, 2]),
            test_train_ratio=1.0),
        data_size=DataSize(ntrain=1, ntest=1))

    # ROC curve of a random classifier
    self.random_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2]),
            test_train_ratio=1.0),
        data_size=DataSize(ntrain=1, ntest=1))

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

  def test_get_result_with_max_positive_predictive_value_first(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(results.get_result_with_max_ppv(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_positive_predictive_value_second(self):
    results = AttackResults(
        [self.random_classifier_result, self.perfect_classifier_result])
    self.assertEqual(results.get_result_with_max_ppv(),
                     self.perfect_classifier_result)

  def test_summary_by_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertSequenceEqual(
        results.summary(by_slices=True),
        'Best-performing attacks over all slices\n' +
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' AUC of 1.00 on slice CORRECTLY_CLASSIFIED=True\n' +
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' advantage of 1.00 on slice CORRECTLY_CLASSIFIED=True\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved a'
        ' positive predictive value of 1.00 on slice CORRECTLY_CLASSIFIED='
        'True\n\n'
        'Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=True"\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' AUC of 1.00\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' advantage of 1.00\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved a'
        ' positive predictive value of 1.00\n\n'
        'Best-performing attacks over slice: "Entire dataset"\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' AUC of 0.50\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' advantage of 0.00\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved a'
        ' positive predictive value of 0.50')

  def test_summary_without_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertSequenceEqual(
        results.summary(by_slices=False),
        'Best-performing attacks over all slices\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' AUC of 1.00 on slice CORRECTLY_CLASSIFIED=True\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved an'
        ' advantage of 1.00 on slice CORRECTLY_CLASSIFIED=True\n'
        '  THRESHOLD_ATTACK (with 1 training and 1 test examples) achieved a'
        ' positive predictive value of 1.00 on slice CORRECTLY_CLASSIFIED=True')

  def test_save_load(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])

    with tempfile.TemporaryDirectory() as tmpdirname:
      filepath = os.path.join(tmpdirname, 'results.pickle')
      results.save(filepath)
      loaded_results = AttackResults.load(filepath)

    self.assertEqual(repr(results), repr(loaded_results))

  def test_calculate_pd_dataframe(self):
    single_results = [
        self.perfect_classifier_result, self.random_classifier_result
    ]
    results = AttackResults(single_results)
    df = results.calculate_pd_dataframe()
    df_expected = pd.DataFrame({
        'slice feature': ['correctly_classified', 'Entire dataset'],
        'slice value': ['True', ''],
        'train size': [1, 1],
        'test size': [1, 1],
        'attack type': ['THRESHOLD_ATTACK', 'THRESHOLD_ATTACK'],
        'Attacker advantage': [1.0, 0.0],
        'Positive predictive value': [1.0, 0.5],
        'AUC': [1.0, 0.5]
    })
    pd.testing.assert_frame_equal(df, df_expected)


if __name__ == '__main__':
  absltest.main()

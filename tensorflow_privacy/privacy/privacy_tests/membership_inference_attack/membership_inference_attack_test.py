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

from absl.testing import absltest
import numpy as np

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import DataSize
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec


def get_test_input(n_train, n_test):
  """Get example inputs for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      logits_train=rng.randn(n_train, 5) + 0.2,
      logits_test=rng.randn(n_test, 5) + 0.2,
      labels_train=np.array([i % 5 for i in range(n_train)]),
      labels_test=np.array([i % 5 for i in range(n_test)]))


def get_multihot_labels_for_test(num_samples: int,
                                 num_classes: int) -> np.ndarray:
  """Generate a array of multihot labels.

  Given an integer 'num_samples', generate a deterministic array of
  'num_classes'multihot labels. Each multihot label is the list of bits (0/1) of
  the corresponding row number in the array, upto 'num_classes'. If the value
  of num_classes < num_samples, then the bit list repeats.
  e.g. if num_samples=10 and num_classes=3, row=3 corresponds to the label
  vector [0, 1, 1].

  Args:
    num_samples: Number of samples for which to generate test labels.
    num_classes: Number of classes for which to generate test multihot labels.

  Returns:
    Numpy integer array with rows=number of samples, and columns=length of the
      bit-representation of the number of samples.
  """
  m = 2**num_classes  # Number of unique labels given the number of classes.
  bit_format = f'0{num_classes}b'  # Bit representation format with leading 0s.
  return np.asarray(
      [list(format(i % m, bit_format)) for i in range(num_samples)]).astype(int)


def get_multilabel_test_input(n_train, n_test):
  """Get example multilabel inputs for attacks."""
  rng = np.random.RandomState(4)
  num_classes = max(n_train // 20, 5)  # use at least 5 classes.
  return AttackInputData(
      logits_train=rng.randn(n_train, num_classes) + 0.2,
      logits_test=rng.randn(n_test, num_classes) + 0.2,
      labels_train=get_multihot_labels_for_test(n_train, num_classes),
      labels_test=get_multihot_labels_for_test(n_test, num_classes))


def get_test_input_logits_only(n_train, n_test):
  """Get example input logits for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      logits_train=rng.randn(n_train, 5) + 0.2,
      logits_test=rng.randn(n_test, 5) + 0.2)


class RunAttacksTest(absltest.TestCase):

  def test_run_attacks_size(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(),
        (AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION))

    self.assertLen(result.single_attack_results, 2)

  def test_trained_attacks_logits_only_size(self):
    result = mia.run_attacks(
        get_test_input_logits_only(100, 100), SlicingSpec(),
        (AttackType.LOGISTIC_REGRESSION,))

    self.assertLen(result.single_attack_results, 1)

  def test_run_attack_trained_sets_attack_type(self):
    result = mia._run_attack(
        get_test_input(100, 100), AttackType.LOGISTIC_REGRESSION)

    self.assertEqual(result.attack_type, AttackType.LOGISTIC_REGRESSION)

  def test_run_attack_threshold_sets_attack_type(self):
    result = mia._run_attack(
        get_test_input(100, 100), AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.attack_type, AttackType.THRESHOLD_ATTACK)

  def test_run_attack_threshold_entropy_sets_attack_type(self):
    result = mia._run_attack(
        get_test_input(100, 100), AttackType.THRESHOLD_ENTROPY_ATTACK)

    self.assertEqual(result.attack_type, AttackType.THRESHOLD_ENTROPY_ATTACK)

  def test_run_attack_threshold_sets_membership_scores(self):
    result = mia._run_attack(
        get_test_input(100, 50), AttackType.THRESHOLD_ATTACK)

    self.assertLen(result.membership_scores_train, 100)
    self.assertLen(result.membership_scores_test, 50)

  def test_run_attack_threshold_entropy_sets_membership_scores(self):
    result = mia._run_attack(
        get_test_input(100, 50), AttackType.THRESHOLD_ENTROPY_ATTACK)

    self.assertLen(result.membership_scores_train, 100)
    self.assertLen(result.membership_scores_test, 50)

  def test_run_attack_trained_sets_membership_scores(self):
    attack_input = AttackInputData(
        logits_train=np.tile([500., -500.], (100, 1)),
        logits_test=np.tile([0., 0.], (50, 1)))

    result = mia._run_trained_attack(
        attack_input,
        AttackType.LOGISTIC_REGRESSION,
        balance_attacker_training=True)
    self.assertLen(result.membership_scores_train, 100)
    self.assertLen(result.membership_scores_test, 50)

    # Scores for all training (resp. test) examples should be close
    np.testing.assert_allclose(
        result.membership_scores_train,
        result.membership_scores_train[0],
        rtol=1e-3)
    np.testing.assert_allclose(
        result.membership_scores_test,
        result.membership_scores_test[0],
        rtol=1e-3)
    # Training score should be smaller than test score
    self.assertLess(result.membership_scores_train[0],
                    result.membership_scores_test[0])

  def test_run_attack_threshold_calculates_correct_auc(self):
    result = mia._run_attack(
        AttackInputData(
            loss_train=np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6]),
            loss_test=np.array([1.1, 1.2, 1.3, 0.4, 1.5, 1.6])),
        AttackType.THRESHOLD_ATTACK)

    np.testing.assert_almost_equal(result.roc_curve.get_auc(), 0.83, decimal=2)

  def test_run_attack_threshold_entropy_calculates_correct_auc(self):
    result = mia._run_attack(
        AttackInputData(
            entropy_train=np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6]),
            entropy_test=np.array([1.1, 1.2, 1.3, 0.4, 1.5, 1.6])),
        AttackType.THRESHOLD_ENTROPY_ATTACK)

    np.testing.assert_almost_equal(result.roc_curve.get_auc(), 0.83, decimal=2)

  def test_run_attack_by_slice(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(by_class=True),
        (AttackType.THRESHOLD_ATTACK,))

    self.assertLen(result.single_attack_results, 6)
    expected_slice = SingleSliceSpec(SlicingFeature.CLASS, 2)
    self.assertEqual(result.single_attack_results[3].slice_spec, expected_slice)

  def test_accuracy(self):
    predictions = [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]
    logits = [[1, -1, -3], [-3, -1, -2], [9, 8, 8.5]]
    labels = [0, 1, 2]
    self.assertEqual(mia._get_accuracy(predictions, labels), 2 / 3)
    self.assertEqual(mia._get_accuracy(logits, labels), 2 / 3)
    # If accuracy is already present, simply return it.
    self.assertIsNone(mia._get_accuracy(None, labels))

  def test_run_compute_membership_probability_correct_probs(self):
    result = mia._compute_membership_probability(
        AttackInputData(
            loss_train=np.array([1, 1, 1, 10, 100]),
            loss_test=np.array([10, 100, 100, 1000, 10000])))

    np.testing.assert_almost_equal(
        result.train_membership_probs, [1, 1, 1, 0.5, 0.33], decimal=2)
    np.testing.assert_almost_equal(
        result.test_membership_probs, [0.5, 0.33, 0.33, 0, 0], decimal=2)

  def test_run_attack_data_size(self):
    result = mia.run_attacks(
        get_test_input(100, 80), SlicingSpec(by_class=True),
        (AttackType.THRESHOLD_ATTACK,))
    self.assertEqual(result.single_attack_results[0].data_size,
                     DataSize(ntrain=100, ntest=80))
    self.assertEqual(result.single_attack_results[3].data_size,
                     DataSize(ntrain=20, ntest=16))


class RunAttacksTestOnMultilabelData(absltest.TestCase):

  def test_run_attacks_size(self):
    result = mia.run_attacks(
        get_multilabel_test_input(100, 100), SlicingSpec(),
        (AttackType.LOGISTIC_REGRESSION,))

    self.assertLen(result.single_attack_results, 1)

  def test_run_attack_trained_sets_attack_type(self):
    result = mia._run_attack(
        get_multilabel_test_input(100, 100), AttackType.LOGISTIC_REGRESSION)

    self.assertEqual(result.attack_type, AttackType.LOGISTIC_REGRESSION)

  def test_run_attack_threshold_sets_attack_type(self):
    result = mia._run_attack(
        get_multilabel_test_input(100, 100), AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.attack_type, AttackType.THRESHOLD_ATTACK)

  def test_run_attack_threshold_entropy_fails(self):
    self.assertRaises(NotImplementedError, mia._run_threshold_entropy_attack,
                      get_multilabel_test_input(100, 100))

  def test_run_attack_by_percentiles_slice(self):
    result = mia.run_attacks(
        get_multilabel_test_input(100, 100),
        SlicingSpec(entire_dataset=True, by_class=False, by_percentiles=True),
        (AttackType.THRESHOLD_ATTACK,))

    # 1 attack on entire dataset, 1 attack each of 10 percentile ranges, total
    # of 11.
    self.assertLen(result.single_attack_results, 11)
    expected_slice = SingleSliceSpec(SlicingFeature.PERCENTILE, (20, 30))
    # First slice (Slice #0) is entire dataset. Hence Slice #3 is the 3rd
    # percentile range 20-30.
    self.assertEqual(result.single_attack_results[3].slice_spec, expected_slice)

  def test_numpy_multilabel_accuracy(self):
    predictions = [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]
    labels = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    # At a threshold=0.5, 5 of the total 9 lables are correct.
    self.assertAlmostEqual(
        mia._get_numpy_binary_accuracy(predictions, labels), 5 / 9, places=6)

  def test_multilabel_accuracy(self):
    predictions = [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]
    labels = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    # At a threshold=0.5, 5 of the total 9 lables are correct.
    self.assertAlmostEqual(
        mia._get_multilabel_accuracy(predictions, labels), 5 / 9, places=6)
    self.assertIsNone(mia._get_accuracy(None, labels))


if __name__ == '__main__':
  absltest.main()

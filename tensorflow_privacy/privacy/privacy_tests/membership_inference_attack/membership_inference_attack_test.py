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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_privacy.privacy.privacy_tests import epsilon_lower_bound as elb
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import models
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


def get_multilabel_test_input_with_sample_weights(n_train, n_test):
  """Get example multilabel inputs for attacks."""
  rng = np.random.RandomState(4)
  num_classes = max(n_train // 20, 5)  # use at least 5 classes.
  return AttackInputData(
      logits_train=rng.randn(n_train, num_classes) + 0.2,
      logits_test=rng.randn(n_test, num_classes) + 0.2,
      labels_train=get_multihot_labels_for_test(n_train, num_classes),
      labels_test=get_multihot_labels_for_test(n_test, num_classes),
      sample_weight_train=rng.randn(n_train, 1),
      sample_weight_test=rng.randn(n_test, 1))


def get_test_input_logits_only(n_train, n_test):
  """Get example input logits for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      logits_train=rng.randn(n_train, 5) + 0.2,
      logits_test=rng.randn(n_test, 5) + 0.2)


def get_test_input_logits_only_with_sample_weights(n_train, n_test):
  """Get example input logits for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      logits_train=rng.randn(n_train, 5) + 0.2,
      logits_test=rng.randn(n_test, 5) + 0.2,
      sample_weight_train=rng.randn(n_train, 1),
      sample_weight_test=rng.randn(n_test, 1))


class MockTrainedAttacker(object):
  """Mock for TrainedAttacker."""

  def __init__(self, backend):
    del backend
    return

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    del input_features, is_training_labels, sample_weight
    return

  def predict(self, input_features):
    return input_features[:, 0]


class RunAttacksTest(parameterized.TestCase):

  def test_run_attacks_size(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(),
        (AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION))

    self.assertLen(result.single_attack_results, 2)

  @parameterized.named_parameters(
      ('low_ratio', 100, 10),
      ('ratio_1', 100, 100),
      ('high_ratio', 100, 1000),
  )
  def test_test_train_ratio(self, ntrain, ntest):
    test_input = get_test_input(ntrain, ntest)
    expected_test_train_ratio = ntest / ntrain
    calculated_test_train_ratio = (
        test_input.get_test_size() / test_input.get_train_size())

    self.assertEqual(expected_test_train_ratio, calculated_test_train_ratio)

  def test_run_attacks_parallel_backend(self):
    result = mia.run_attacks(
        get_multilabel_test_input(100, 100),
        SlicingSpec(), (
            AttackType.THRESHOLD_ATTACK,
            AttackType.LOGISTIC_REGRESSION,
        ),
        backend='threading')

    self.assertLen(result.single_attack_results, 2)

  def test_run_attacks_parallel_backend_with_sample_weights(self):
    result = mia.run_attacks(
        get_multilabel_test_input_with_sample_weights(100, 100),
        SlicingSpec(), (
            AttackType.LOGISTIC_REGRESSION,
            AttackType.RANDOM_FOREST,
        ),
        backend='threading')

    self.assertLen(result.single_attack_results, 2)

  def test_trained_attacks_logits_only_size(self):
    result = mia.run_attacks(
        get_test_input_logits_only(100, 100), SlicingSpec(),
        (AttackType.LOGISTIC_REGRESSION,))

    self.assertLen(result.single_attack_results, 1)

  def test_trained_attacks_logits_only_with_sample_weights_size(self):
    result = mia.run_attacks(
        get_test_input_logits_only_with_sample_weights(100, 100), SlicingSpec(),
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
    # Training score should be larger than test score, as training set is set
    # to be positive.
    self.assertGreater(
        result.membership_scores_train[0], result.membership_scores_test[0]
    )

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

  @parameterized.parameters(
      [AttackType.THRESHOLD_ATTACK],
      [AttackType.THRESHOLD_ENTROPY_ATTACK],
  )
  def test_calculates_correct_tpr_fpr(self, attack_type):
    rng = np.random.RandomState(27)
    loss_train = rng.rand(100)
    loss_test = rng.rand(50) + 0.1
    result = mia._run_attack(
        AttackInputData(
            loss_train=loss_train,
            loss_test=loss_test,
            entropy_train=loss_train,
            entropy_test=loss_test,
        ),
        attack_type,
    )
    self.assertEqual(attack_type, result.attack_type)
    for tpr, fpr, threshold in zip(
        result.roc_curve.tpr, result.roc_curve.fpr, result.roc_curve.thresholds
    ):
      self.assertAlmostEqual(tpr, np.mean(loss_train <= threshold))
      self.assertAlmostEqual(fpr, np.mean(loss_test <= threshold))

  @mock.patch('sklearn.metrics.roc_curve')
  def test_run_attack_threshold_entropy_small_tpr_fpr_correct_ppv(
      self, patched_fn):
    # sklearn.metrics.roc_curve returns (fpr, tpr, thresholds).
    patched_fn.return_value = (
        np.array([0.2, 0.04, 0.0003]),
        np.array([0.1, 0.0001, 0.0002]),
        np.array([0.2, 0.4, 0.6]),
    )
    result = mia._run_attack(
        AttackInputData(
            entropy_train=np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6]),
            entropy_test=np.array([1.1, 1.2, 1.3, 0.4, 1.5, 1.6]),
            force_multilabel_data=False), AttackType.THRESHOLD_ENTROPY_ATTACK)
    # PPV = TPR / (TPR + test_train_ratio * FPR), except when both TPR and FPR
    # are close to 0. Then PPV = 1/ (1 + test_train_ratio)
    # With the above values, TPR / (TPR + test_train_ratio * FPR) =
    # 0.1 / (0.1 + (6/6) * 0.2) = 0.333,
    # 0.0001 / (0.0001 + (6/6) * 0.04) = 0.002493,
    # and 1/ (1+ (6/6)) = 0.5. So PPV is the max of these three values,
    # namely 0.5.
    np.testing.assert_almost_equal(result.roc_curve.get_ppv(), 0.5, decimal=2)

  @parameterized.parameters(
      (AttackType.THRESHOLD_ATTACK, 'loss_train', 'loss_test'),
      (AttackType.THRESHOLD_ENTROPY_ATTACK, 'entropy_train', 'entropy_test'),
  )
  @mock.patch.object(mia, 'EPSILON_K', 4)
  def test_run_attack_threshold_calculates_correct_epsilon(
      self, attack_type, train_metric_name, test_metric_name
  ):
    result = mia._run_attack(
        AttackInputData(
            **{
                train_metric_name: np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6]),
                test_metric_name: np.array([1.1, 1.2, 1.3, 0.4, 1.5, 1.6]),
            }
        ),
        attack_type,
    )
    np.testing.assert_almost_equal(
        result.epsilon_lower_bound_value.get_max_epsilon_bounds(),
        np.array([0.34695111, 0.34695111, 0.05616349, 0.05616349]),
    )

  @parameterized.product(
      (
          dict(
              epsilon_methods=tuple(elb.BoundMethod),
              expected_max_method=elb.BoundMethod.CLOPPER_PEARSON,
          ),
          dict(
              epsilon_methods=(
                  elb.BoundMethod.KATZ_LOG,
                  elb.BoundMethod.CLOPPER_PEARSON,
              ),
              expected_max_method=elb.BoundMethod.CLOPPER_PEARSON,
          ),
          dict(
              epsilon_methods=(elb.BoundMethod.BAILEY,),
              expected_max_method=elb.BoundMethod.BAILEY,
          ),
      ),
      attack_type=[
          AttackType.LOGISTIC_REGRESSION,
          AttackType.MULTI_LAYERED_PERCEPTRON,
          AttackType.RANDOM_FOREST,
          AttackType.K_NEAREST_NEIGHBORS,
      ],
  )
  @mock.patch.object(models, 'TrainedAttacker', MockTrainedAttacker)
  @mock.patch.object(models, 'LogisticRegressionAttacker', MockTrainedAttacker)
  @mock.patch.object(
      models, 'MultilayerPerceptronAttacker', MockTrainedAttacker
  )
  @mock.patch.object(models, 'RandomForestAttacker', MockTrainedAttacker)
  @mock.patch.object(models, 'KNearestNeighborsAttacker', MockTrainedAttacker)
  def test_run_attack_trained_calculates_correct_epsilon(
      self,
      epsilon_methods,
      expected_max_method,
      attack_type,
  ):
    logits_train = np.ones((1000, 5))
    logits_test = np.zeros((100, 5))
    # The prediction would be all 1 for training and all 0 for test.
    expected_bounds = {
        elb.BoundMethod.KATZ_LOG: [
            5.27530977,
            2.97796578,
            -0.00720554,
            -0.01623037,
        ],
        elb.BoundMethod.ADJUSTED_LOG: [
            5.27580935,
            2.98292432,
            -0.00717236,
            -0.01614769,
            -7.62368550,
        ],
        elb.BoundMethod.BAILEY: [
            5.87410287,
            3.57903543,
            -0.00718316,
            -0.01625286,
        ],
        elb.BoundMethod.INV_SINH: [
            4.95123977,
            2.65964282,
            -0.00720547,
            -0.01623030,
        ],
        elb.BoundMethod.CLOPPER_PEARSON: [
            5.56738762,
            3.31454626,
        ],
    }
    with mock.patch.object(mia, 'EPSILON_METHODS', epsilon_methods):
      result = mia._run_attack(
          AttackInputData(logits_train, logits_test),
          attack_type,
      )
      self.assertCountEqual(
          result.epsilon_lower_bound_value.bounds.keys(), epsilon_methods
      )
      for key in epsilon_methods:
        np.testing.assert_almost_equal(
            result.epsilon_lower_bound_value.bounds[key], expected_bounds[key]
        )
      np.testing.assert_almost_equal(
          result.epsilon_lower_bound_value.get_max_epsilon_bounds(),
          expected_bounds[expected_max_method],
      )

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

  def test_run_attacks_parallel_backend(self):
    result = mia.run_attacks(
        get_multilabel_test_input(100, 100),
        SlicingSpec(), (AttackType.LOGISTIC_REGRESSION,),
        backend='threading')

    self.assertLen(result.single_attack_results, 1)

  def test_run_attacks_parallel_backend_with_sample_weights(self):
    result = mia.run_attacks(
        get_multilabel_test_input_with_sample_weights(100, 100),
        SlicingSpec(),
        (AttackType.LOGISTIC_REGRESSION, AttackType.RANDOM_FOREST),
        backend='threading')

    self.assertLen(result.single_attack_results, 2)

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

  def test_run_multilabel_attack_threshold_calculates_correct_ppv(self):
    result = mia._run_attack(
        AttackInputData(
            loss_train=np.array([[0.1, 0.2], [1.3, 0.4], [0.5, 0.6], [0.9,
                                                                      0.6]]),
            loss_test=np.array([[1.1, 1.2], [1.3, 0.4], [1.5, 1.6]]),
            force_multilabel_data=True), AttackType.THRESHOLD_ATTACK)

    np.testing.assert_almost_equal(result.roc_curve.get_ppv(), 1.0, decimal=2)

  @mock.patch('sklearn.metrics.roc_curve')
  def test_run_multilabel_attack_threshold_small_tpr_fpr_correct_ppv(
      self, patched_fn):
    # sklearn.metrics.roc_curve returns (fpr, tpr, thresholds).
    patched_fn.return_value = (
        np.array([0.2, 0.04, 0.0003]),
        np.array([0.1, 0.0001, 0.0002]),
        np.array([0.2, 0.4, 0.6]),
    )
    result = mia._run_attack(
        AttackInputData(
            loss_train=np.array([[0.1, 0.2], [1.3, 0.4], [0.5, 0.6], [0.9,
                                                                      0.6]]),
            loss_test=np.array([[1.1, 1.2], [1.3, 0.4], [1.5, 1.6]]),
            force_multilabel_data=True), AttackType.THRESHOLD_ATTACK)
    # PPV = TPR / (TPR + test_train_ratio * FPR), except when both TPR and FPR
    # are close to 0. Then PPV = 1/ (1 + test_train_ratio)
    # With the above values, TPR / (TPR + test_train_ratio * FPR) =
    # 0.1 / (0.1 + (3/4) * 0.2) = 0.4,
    # 0.0001 / (0.0001 + (3/4) * 0.04) = 0.003322,
    # and 1/ (1+ 0.75) = 0.57142. So PPV is the max of these three values,
    # namely 0.57142.
    np.testing.assert_almost_equal(
        result.roc_curve.get_ppv(), 0.57142, decimal=2)

  def test_run_attacks_insufficient_examples(self):
    result = mia.run_attacks(
        get_test_input(1, 100),
        SlicingSpec(),
        (
            AttackType.THRESHOLD_ATTACK,
            AttackType.LOGISTIC_REGRESSION,
        ),
    )
    self.assertLen(result.single_attack_results, 1)
    self.assertEqual(
        result.single_attack_results[0].attack_type.value,
        AttackType.THRESHOLD_ATTACK.value,
    )

  def test_run_attacks_size_return_indices(self):
    result = mia.run_attacks(
        get_test_input(100, 100),
        SlicingSpec(
            entire_dataset=False,
            by_class=True,
            by_percentiles=True,
            by_classification_correctness=True,
        ),
        (AttackType.LOGISTIC_REGRESSION,),
        return_slice_indices=True,
    )

    for attack_result in result.single_attack_results:
      self.assertIsNotNone(attack_result.train_indices)
      self.assertIsNotNone(attack_result.test_indices)


if __name__ == '__main__':
  absltest.main()

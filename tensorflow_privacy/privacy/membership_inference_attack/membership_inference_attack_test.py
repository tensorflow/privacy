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
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import Seq2SeqAttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec


def get_test_input(n_train, n_test):
  """Get example inputs for attacks."""
  rng = np.random.RandomState(4)
  return AttackInputData(
      logits_train=rng.randn(n_train, 5) + 0.2,
      logits_test=rng.randn(n_test, 5) + 0.2,
      labels_train=np.array([i % 5 for i in range(n_train)]),
      labels_test=np.array([i % 5 for i in range(n_test)]))


def get_seq2seq_test_input(n_train, n_test, max_seq_in_batch, max_tokens_in_sequence, vocab_size, seed=None):
  """Returns example inputs for attacks on seq2seq models."""
  if seed is not None:
    np.random.seed(seed=seed)

  logits_train, labels_train = [], []
  for i in range(n_train):
    num_sequences = np.random.choice(max_seq_in_batch, 1)[0] + 1
    batch_logits, batch_labels = _get_batch_logits_and_labels(num_sequences, max_tokens_in_sequence, vocab_size)
    logits_train.append(batch_logits)
    labels_train.append(batch_labels)

  logits_test, labels_test = [], []
  for i in range(n_test):
    num_sequences = np.random.choice(max_seq_in_batch, 1)[0] + 1
    batch_logits, batch_labels = _get_batch_logits_and_labels(num_sequences, max_tokens_in_sequence, vocab_size)
    logits_test.append(batch_logits)
    labels_test.append(batch_labels)

  return Seq2SeqAttackInputData(
    logits_train=iter(logits_train),
    logits_test=iter(logits_test),
    labels_train=iter(labels_train),
    labels_test=iter(labels_test),
    vocab_size=vocab_size,
    train_size=n_train,
    test_size=n_test
  )


def _get_batch_logits_and_labels(num_sequences, max_tokens_in_sequence, vocab_size):
  num_tokens_in_sequence = np.random.choice(max_tokens_in_sequence, num_sequences) + 1
  batch_logits, batch_labels = [], []
  for num_tokens in num_tokens_in_sequence:
    logits, labels = _get_sequence_logits_and_labels(num_tokens, vocab_size)
    batch_logits.append(logits)
    batch_labels.append(labels)
  return np.array(batch_logits, dtype=object), np.array(batch_labels, dtype=object)


def _get_sequence_logits_and_labels(num_tokens, vocab_size):
  sequence_logits = []
  for i in range(num_tokens):
    token_logits = np.random.random(vocab_size)
    token_logits /= token_logits.sum()
    sequence_logits.append(token_logits)
  sequence_labels = np.random.choice(vocab_size, num_tokens)
  return np.array(sequence_logits, dtype=np.float32), np.array(sequence_labels, dtype=np.float32)


class RunAttacksTest(absltest.TestCase):

  def test_run_attacks_size(self):
    result = mia.run_attacks(
        get_test_input(100, 100), SlicingSpec(),
        (AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION))

    self.assertLen(result.single_attack_results, 2)

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

  def test_run_seq2seq_attack_size(self):
    result = mia.run_seq2seq_attack(
      get_seq2seq_test_input(n_train=10, n_test=5,
                             max_seq_in_batch=3,
                             max_tokens_in_sequence=5,
                             vocab_size=2))

    self.assertLen(result.single_attack_results, 1)

  def test_run_seq2seq_attack_trained_sets_attack_type(self):
    result = mia.run_seq2seq_attack(
      get_seq2seq_test_input(n_train=10, n_test=5,
                             max_seq_in_batch=3,
                             max_tokens_in_sequence=5,
                             vocab_size=2))
    seq2seq_result = list(result.single_attack_results)[0]
    self.assertEqual(seq2seq_result.attack_type, AttackType.LOGISTIC_REGRESSION)

  def test_run_seq2seq_attack_calculates_correct_auc(self):
    result = mia.run_seq2seq_attack(
      get_seq2seq_test_input(n_train=20, n_test=10,
                             max_seq_in_batch=3,
                             max_tokens_in_sequence=5,
                             vocab_size=3, seed=12345),
      balance_attacker_training=False)
    seq2seq_result = list(result.single_attack_results)[0]
    np.testing.assert_almost_equal(seq2seq_result.roc_curve.get_auc(), 0.63, decimal=2)


if __name__ == '__main__':
  absltest.main()

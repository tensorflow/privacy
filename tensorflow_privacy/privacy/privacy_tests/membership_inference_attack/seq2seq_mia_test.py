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

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.seq2seq_mia import run_seq2seq_attack
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.seq2seq_mia import Seq2SeqAttackInputData


class Seq2SeqAttackInputDataTest(absltest.TestCase):

  def test_validator(self):
    valid_logits_train = iter([np.array([]), np.array([])])
    valid_logits_test = iter([np.array([]), np.array([])])
    valid_labels_train = iter([np.array([]), np.array([])])
    valid_labels_test = iter([np.array([]), np.array([])])

    invalid_logits_train = []
    invalid_logits_test = []
    invalid_labels_train = []
    invalid_labels_test = []

    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(logits_train=valid_logits_train).validate)
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(labels_train=valid_labels_train).validate)
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(logits_test=valid_logits_test).validate)
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(labels_test=valid_labels_test).validate)
    self.assertRaises(ValueError, Seq2SeqAttackInputData(vocab_size=0).validate)
    self.assertRaises(ValueError, Seq2SeqAttackInputData(train_size=0).validate)
    self.assertRaises(ValueError, Seq2SeqAttackInputData(test_size=0).validate)
    self.assertRaises(ValueError, Seq2SeqAttackInputData().validate)

    # Tests that both logits and labels must be set.
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(
            logits_train=valid_logits_train,
            logits_test=valid_logits_test,
            vocab_size=0,
            train_size=0,
            test_size=0).validate)
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(
            labels_train=valid_labels_train,
            labels_test=valid_labels_test,
            vocab_size=0,
            train_size=0,
            test_size=0).validate)

    # Tests that vocab, train, test sizes must all be set.
    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(
            logits_train=valid_logits_train,
            logits_test=valid_logits_test,
            labels_train=valid_labels_train,
            labels_test=valid_labels_test).validate)

    self.assertRaises(
        ValueError,
        Seq2SeqAttackInputData(
            logits_train=invalid_logits_train,
            logits_test=invalid_logits_test,
            labels_train=invalid_labels_train,
            labels_test=invalid_labels_test,
            vocab_size=0,
            train_size=0,
            test_size=0).validate)


def _get_batch_logits_and_labels(num_sequences, max_tokens_in_sequence,
                                 vocab_size):
  num_tokens_in_sequence = np.random.choice(max_tokens_in_sequence,
                                            num_sequences) + 1
  batch_logits, batch_labels = [], []
  for num_tokens in num_tokens_in_sequence:
    logits, labels = _get_sequence_logits_and_labels(num_tokens, vocab_size)
    batch_logits.append(logits)
    batch_labels.append(labels)
  return np.array(
      batch_logits, dtype=object), np.array(
          batch_labels, dtype=object)


def _get_sequence_logits_and_labels(num_tokens, vocab_size):
  sequence_logits = []
  for _ in range(num_tokens):
    token_logits = np.random.random(vocab_size)
    token_logits /= token_logits.sum()
    sequence_logits.append(token_logits)
  sequence_labels = np.random.choice(vocab_size, num_tokens)
  return np.array(
      sequence_logits, dtype=np.float32), np.array(
          sequence_labels, dtype=np.float32)


def get_seq2seq_test_input(n_train,
                           n_test,
                           max_seq_in_batch,
                           max_tokens_in_sequence,
                           vocab_size,
                           seed=None):
  """Returns example inputs for attacks on seq2seq models."""
  if seed is not None:
    np.random.seed(seed=seed)

  logits_train, labels_train = [], []
  for _ in range(n_train):
    num_sequences = np.random.choice(max_seq_in_batch, 1)[0] + 1
    batch_logits, batch_labels = _get_batch_logits_and_labels(
        num_sequences, max_tokens_in_sequence, vocab_size)
    logits_train.append(batch_logits)
    labels_train.append(batch_labels)

  logits_test, labels_test = [], []
  for _ in range(n_test):
    num_sequences = np.random.choice(max_seq_in_batch, 1)[0] + 1
    batch_logits, batch_labels = _get_batch_logits_and_labels(
        num_sequences, max_tokens_in_sequence, vocab_size)
    logits_test.append(batch_logits)
    labels_test.append(batch_labels)

  return Seq2SeqAttackInputData(
      logits_train=iter(logits_train),
      logits_test=iter(logits_test),
      labels_train=iter(labels_train),
      labels_test=iter(labels_test),
      vocab_size=vocab_size,
      train_size=n_train,
      test_size=n_test)


class RunSeq2SeqAttackTest(absltest.TestCase):

  def test_run_seq2seq_attack_size(self):
    result = run_seq2seq_attack(
        get_seq2seq_test_input(
            n_train=10,
            n_test=5,
            max_seq_in_batch=3,
            max_tokens_in_sequence=5,
            vocab_size=2))

    self.assertLen(result.single_attack_results, 1)

  def test_run_seq2seq_attack_trained_sets_attack_type(self):
    result = run_seq2seq_attack(
        get_seq2seq_test_input(
            n_train=10,
            n_test=5,
            max_seq_in_batch=3,
            max_tokens_in_sequence=5,
            vocab_size=2))
    seq2seq_result = list(result.single_attack_results)[0]
    self.assertEqual(seq2seq_result.attack_type, AttackType.THRESHOLD_ATTACK)

  def test_run_seq2seq_attack_calculates_correct_auc(self):
    result = run_seq2seq_attack(
        get_seq2seq_test_input(
            n_train=20,
            n_test=10,
            max_seq_in_batch=3,
            max_tokens_in_sequence=5,
            vocab_size=3,
            seed=12345),
        balance_attacker_training=False)
    seq2seq_result = list(result.single_attack_results)[0]
    np.testing.assert_almost_equal(
        seq2seq_result.roc_curve.get_auc(), 0.59, decimal=2)

  def test_run_seq2seq_attack_calculates_correct_metadata(self):
    attack_input = Seq2SeqAttackInputData(
        logits_train=iter([
            np.array([
                np.array([[0.1, 0.1, 0.8], [0.7, 0.3, 0]], dtype=np.float32),
                np.array([[0.4, 0.5, 0.1]], dtype=np.float32)
            ],
                     dtype=object),
            np.array(
                [np.array([[0.25, 0.6, 0.15], [1, 0, 0]], dtype=np.float32)],
                dtype=object),
            np.array([
                np.array([[0.9, 0, 0.1], [0.25, 0.5, 0.25]], dtype=np.float32),
                np.array([[0, 1, 0], [0.2, 0.1, 0.7]], dtype=np.float32)
            ],
                     dtype=object),
            np.array([
                np.array([[0.9, 0, 0.1], [0.25, 0.5, 0.25]], dtype=np.float32),
                np.array([[0, 1, 0], [0.2, 0.1, 0.7]], dtype=np.float32)
            ],
                     dtype=object)
        ]),
        logits_test=iter([
            np.array([
                np.array([[0.25, 0.4, 0.35], [0.2, 0.4, 0.4]], dtype=np.float32)
            ],
                     dtype=object),
            np.array([
                np.array([[0.3, 0.3, 0.4], [0.4, 0.4, 0.2]], dtype=np.float32),
                np.array([[0.3, 0.35, 0.35]], dtype=np.float32)
            ],
                     dtype=object),
            np.array([
                np.array([[0.25, 0.4, 0.35], [0.2, 0.4, 0.4]], dtype=np.float32)
            ],
                     dtype=object),
            np.array([
                np.array([[0.25, 0.4, 0.35], [0.2, 0.4, 0.4]], dtype=np.float32)
            ],
                     dtype=object)
        ]),
        labels_train=iter([
            np.array([
                np.array([2, 0], dtype=np.float32),
                np.array([1], dtype=np.float32)
            ],
                     dtype=object),
            np.array([np.array([1, 0], dtype=np.float32)], dtype=object),
            np.array([
                np.array([0, 1], dtype=np.float32),
                np.array([1, 2], dtype=np.float32)
            ],
                     dtype=object),
            np.array([
                np.array([0, 0], dtype=np.float32),
                np.array([0, 1], dtype=np.float32)
            ],
                     dtype=object)
        ]),
        labels_test=iter([
            np.array([np.array([2, 1], dtype=np.float32)]),
            np.array([
                np.array([2, 0], dtype=np.float32),
                np.array([1], dtype=np.float32)
            ],
                     dtype=object),
            np.array([np.array([2, 1], dtype=np.float32)]),
            np.array([np.array([2, 1], dtype=np.float32)]),
        ]),
        vocab_size=3,
        train_size=4,
        test_size=4)
    result = run_seq2seq_attack(attack_input, balance_attacker_training=False)
    metadata = result.privacy_report_metadata
    np.testing.assert_almost_equal(metadata.loss_train, 0.91, decimal=2)
    np.testing.assert_almost_equal(metadata.loss_test, 1.58, decimal=2)
    np.testing.assert_almost_equal(metadata.accuracy_train, 0.77, decimal=2)
    np.testing.assert_almost_equal(metadata.accuracy_test, 0.67, decimal=2)


if __name__ == '__main__':
  absltest.main()

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
"""Code for membership inference attacks on seq2seq models.

Contains seq2seq specific logic for attack data structures, attack data
generation,
and the logistic regression membership inference attack.
"""
from typing import Iterator, List

from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import models
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import DataSize
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.models import _sample_multidimensional_array
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.models import AttackerData


def _is_iterator(obj, obj_name):
  """Checks whether obj is a generator."""
  if obj is not None and not isinstance(obj, Iterator):
    raise ValueError('%s should be a generator.' % obj_name)


@dataclass
class Seq2SeqAttackInputData:
  """Input data for running an attack on seq2seq models.

  This includes only the data, and not configuration.
  """
  logits_train: Iterator[np.ndarray] = None
  logits_test: Iterator[np.ndarray] = None

  # Contains ground-truth token indices for the target sequences.
  labels_train: Iterator[np.ndarray] = None
  labels_test: Iterator[np.ndarray] = None

  # Size of the target sequence vocabulary.
  vocab_size: int = None

  # Train, test size = number of batches in training, test set.
  # These values need to be supplied by the user as logits, labels
  # are lazy loaded for seq2seq models.
  train_size: int = 0
  test_size: int = 0

  def validate(self):
    """Validates the inputs."""

    if (self.logits_train is None) != (self.logits_test is None):
      raise ValueError(
          'logits_train and logits_test should both be either set or unset')

    if (self.labels_train is None) != (self.labels_test is None):
      raise ValueError(
          'labels_train and labels_test should both be either set or unset')

    if self.logits_train is None or self.labels_train is None:
      raise ValueError(
          'Labels, logits of training, test sets should all be set')

    if (self.vocab_size is None or self.train_size is None or
        self.test_size is None):
      raise ValueError('vocab_size, train_size, test_size should all be set')

    if self.vocab_size is not None and not int:
      raise ValueError('vocab_size should be of integer type')

    if self.train_size is not None and not int:
      raise ValueError('train_size should be of integer type')

    if self.test_size is not None and not int:
      raise ValueError('test_size should be of integer type')

    _is_iterator(self.logits_train, 'logits_train')
    _is_iterator(self.logits_test, 'logits_test')
    _is_iterator(self.labels_train, 'labels_train')
    _is_iterator(self.labels_test, 'labels_test')

  def __str__(self):
    """Returns the shapes of variables that are not None."""
    result = ['AttackInputData(']

    if self.vocab_size is not None and self.train_size is not None:
      result.append(
          'logits_train with shape (%d, num_sequences, num_tokens, %d)' %
          (self.train_size, self.vocab_size))
      result.append(
          'labels_train with shape (%d, num_sequences, num_tokens, 1)' %
          self.train_size)

    if self.vocab_size is not None and self.test_size is not None:
      result.append(
          'logits_test with shape (%d, num_sequences, num_tokens, %d)' %
          (self.test_size, self.vocab_size))
      result.append(
          'labels_test with shape (%d, num_sequences, num_tokens, 1)' %
          self.test_size)

    result.append(')')
    return '\n'.join(result)


def _get_attack_features_and_metadata(
    logits: Iterator[np.ndarray],
    labels: Iterator[np.ndarray]) -> (np.ndarray, float, float):
  """Returns the average rank of tokens per batch of sequences and the loss.

  Args:
    logits: Logits returned by a seq2seq model, dim = (num_batches,
      num_sequences, num_tokens, vocab_size).
    labels: Target labels for the seq2seq model, dim = (num_batches,
      num_sequences, num_tokens, 1).

  Returns:
    1. An array of average ranks, dim = (num_batches, 1).
    Each average rank is calculated over ranks of tokens in sequences of a
    particular batch.
    2. Loss computed over all logits and labels.
    3. Accuracy computed over all logits and labels.
  """
  ranks = []
  loss = 0.0
  dataset_length = 0.0
  correct_preds = 0
  total_preds = 0
  for batch_logits, batch_labels in zip(logits, labels):
    # Compute average rank for the current batch.
    batch_ranks = _get_batch_ranks(batch_logits, batch_labels)
    ranks.append(np.mean(batch_ranks))

    # Update overall loss metrics with metrics of the current batch.
    batch_loss, batch_length = _get_batch_loss_metrics(batch_logits,
                                                       batch_labels)
    loss += batch_loss
    dataset_length += batch_length

    # Update overall accuracy metrics with metrics of the current batch.
    batch_correct_preds, batch_total_preds = _get_batch_accuracy_metrics(
        batch_logits, batch_labels)
    correct_preds += batch_correct_preds
    total_preds += batch_total_preds

  # Compute loss and accuracy for the dataset.
  loss = loss / dataset_length
  accuracy = correct_preds / total_preds

  return np.array(ranks), loss, accuracy


def _get_batch_ranks(batch_logits: np.ndarray,
                     batch_labels: np.ndarray) -> np.ndarray:
  """Returns the ranks of tokens in a batch of sequences.

  Args:
    batch_logits: Logits returned by a seq2seq model, dim = (num_sequences,
      num_tokens, vocab_size).
    batch_labels: Target labels for the seq2seq model, dim = (num_sequences,
      num_tokens, 1).

  Returns:
    An array of ranks of tokens in a batch of sequences, dim = (num_sequences,
    num_tokens, 1)
  """
  batch_ranks = []
  for sequence_logits, sequence_labels in zip(batch_logits, batch_labels):
    batch_ranks += _get_ranks_for_sequence(sequence_logits, sequence_labels)

  return np.array(batch_ranks)


def _get_ranks_for_sequence(logits: np.ndarray,
                            labels: np.ndarray) -> List[float]:
  """Returns ranks for a sequence.

  Args:
    logits: Logits of a single sequence, dim = (num_tokens, vocab_size).
    labels: Target labels of a single sequence, dim = (num_tokens, 1).

  Returns:
    An array of ranks for tokens in the sequence, dim = (num_tokens, 1).
  """
  sequence_ranks = []
  for logit, label in zip(logits, labels.astype(int)):
    rank = rankdata(-logit, method='min')[label] - 1.0
    sequence_ranks.append(rank)

  return sequence_ranks


def _get_batch_loss_metrics(batch_logits: np.ndarray,
                            batch_labels: np.ndarray) -> (float, int):
  """Returns the loss, number of sequences for a batch.

  Args:
    batch_logits: Logits returned by a seq2seq model, dim = (num_sequences,
      num_tokens, vocab_size).
    batch_labels: Target labels for the seq2seq model, dim = (num_sequences,
      num_tokens, 1).
  """
  batch_loss = 0.0
  batch_length = len(batch_logits)
  for sequence_logits, sequence_labels in zip(batch_logits, batch_labels):
    sequence_loss = tf.losses.sparse_categorical_crossentropy(
        tf.keras.backend.constant(sequence_labels),
        tf.keras.backend.constant(sequence_logits),
        from_logits=True)
    batch_loss += sequence_loss.numpy().sum()

  return batch_loss / batch_length, batch_length


def _get_batch_accuracy_metrics(batch_logits: np.ndarray,
                                batch_labels: np.ndarray) -> (float, float):
  """Returns the number of correct predictions, total number of predictions for a batch.

  Args:
    batch_logits: Logits returned by a seq2seq model, dim = (num_sequences,
      num_tokens, vocab_size).
    batch_labels: Target labels for the seq2seq model, dim = (num_sequences,
      num_tokens, 1).
  """
  batch_correct_preds = 0.0
  batch_total_preds = 0.0
  for sequence_logits, sequence_labels in zip(batch_logits, batch_labels):
    preds = tf.metrics.sparse_categorical_accuracy(
        tf.keras.backend.constant(sequence_labels),
        tf.keras.backend.constant(sequence_logits))
    batch_correct_preds += preds.numpy().sum()
    batch_total_preds += len(sequence_labels)

  return batch_correct_preds, batch_total_preds


def create_seq2seq_attacker_data(
    attack_input_data: Seq2SeqAttackInputData,
    test_fraction: float = 0.25,
    balance: bool = True,
    privacy_report_metadata: PrivacyReportMetadata = PrivacyReportMetadata()
) -> AttackerData:
  """Prepares Seq2SeqAttackInputData to train ML attackers.

  Uses logits and losses to generate ranks and performs a random train-test
  split.

  Also computes metadata (loss, accuracy) for the model under attack
  and populates respective fields of PrivacyReportMetadata.

  Args:
    attack_input_data: Original Seq2SeqAttackInputData
    test_fraction: Fraction of the dataset to include in the test split.
    balance: Whether the training and test sets for the membership inference
      attacker should have a balanced (roughly equal) number of samples from the
      training and test sets used to develop the model under attack.
    privacy_report_metadata: the metadata of the model under attack.

  Returns:
    AttackerData.
  """
  attack_input_train, loss_train, accuracy_train = _get_attack_features_and_metadata(
      attack_input_data.logits_train, attack_input_data.labels_train)
  attack_input_test, loss_test, accuracy_test = _get_attack_features_and_metadata(
      attack_input_data.logits_test, attack_input_data.labels_test)

  if balance:
    min_size = min(len(attack_input_train), len(attack_input_test))
    attack_input_train = _sample_multidimensional_array(attack_input_train,
                                                        min_size)
    attack_input_test = _sample_multidimensional_array(attack_input_test,
                                                       min_size)

  features_all = np.concatenate((attack_input_train, attack_input_test))
  ntrain, ntest = attack_input_train.shape[0], attack_input_test.shape[0]

  # Reshape for classifying one-dimensional features
  features_all = features_all.reshape(-1, 1)

  labels_all = np.concatenate(((np.zeros(ntrain)), (np.ones(ntest))))

  # Perform a train-test split
  features_train, features_test, \
  is_training_labels_train, is_training_labels_test = \
    model_selection.train_test_split(
        features_all, labels_all, test_size=test_fraction, stratify=labels_all)

  # Populate accuracy, loss fields in privacy report metadata
  privacy_report_metadata.loss_train = loss_train
  privacy_report_metadata.loss_test = loss_test
  privacy_report_metadata.accuracy_train = accuracy_train
  privacy_report_metadata.accuracy_test = accuracy_test

  return AttackerData(features_train, is_training_labels_train, features_test,
                      is_training_labels_test,
                      DataSize(ntrain=ntrain, ntest=ntest))


def run_seq2seq_attack(attack_input: Seq2SeqAttackInputData,
                       privacy_report_metadata: PrivacyReportMetadata = None,
                       balance_attacker_training: bool = True) -> AttackResults:
  """Runs membership inference attacks on a seq2seq model.

  Args:
    attack_input: input data for running an attack
    privacy_report_metadata: the metadata of the model under attack.
    balance_attacker_training: Whether the training and test sets for the
      membership inference attacker should have a balanced (roughly equal)
      number of samples from the training and test sets used to develop the
      model under attack.

  Returns:
    the attack result.
  """
  attack_input.validate()

  # The attacker uses the average rank (a single number) of a seq2seq dataset
  # record to determine membership. So only Logistic Regression is supported,
  # as it makes the most sense for single-number features.
  attacker = models.LogisticRegressionAttacker()

  # Create attacker data and populate fields of privacy_report_metadata
  privacy_report_metadata = privacy_report_metadata or PrivacyReportMetadata()
  prepared_attacker_data = create_seq2seq_attacker_data(
      attack_input_data=attack_input,
      balance=balance_attacker_training,
      privacy_report_metadata=privacy_report_metadata)

  attacker.train_model(prepared_attacker_data.features_train,
                       prepared_attacker_data.is_training_labels_train)

  # Run the attacker on (permuted) test examples.
  predictions_test = attacker.predict(prepared_attacker_data.features_test)

  # Generate ROC curves with predictions.
  fpr, tpr, thresholds = metrics.roc_curve(
      prepared_attacker_data.is_training_labels_test, predictions_test)

  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  attack_results = [
      SingleAttackResult(
          slice_spec=SingleSliceSpec(),
          attack_type=AttackType.LOGISTIC_REGRESSION,
          roc_curve=roc_curve,
          data_size=prepared_attacker_data.data_size)
  ]

  return AttackResults(
      single_attack_results=attack_results,
      privacy_report_metadata=privacy_report_metadata)

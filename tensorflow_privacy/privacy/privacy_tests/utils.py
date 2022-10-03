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
"""Utility functions for membership inference attacks."""

import enum
import logging
from typing import Callable, Optional, Union

import numpy as np
from scipy import special


class LossFunction(enum.Enum):
  """An enum that defines loss function."""
  CROSS_ENTROPY = 'cross_entropy'
  SQUARED = 'squared'


LossFunctionCallable = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]],
                                np.ndarray]
LossFunctionType = Union[LossFunctionCallable, LossFunction, str]


def log_loss(labels: np.ndarray,
             pred: np.ndarray,
             sample_weight: Optional[np.ndarray] = None,
             from_logits=False,
             small_value=1e-8) -> np.ndarray:
  """Computes the per-example cross entropy loss.

  Args:
    labels: numpy array of shape (num_samples,). labels[i] is the true label
      (scalar) of the i-th sample and is one of {0, 1, ..., num_classes-1}.
    pred: numpy array of shape (num_samples, num_classes) or (num_samples,). For
      categorical cross entropy loss, the shape should be (num_samples,
      num_classes) and pred[i] is the logits or probability vector of the i-th
      sample. For binary logistic loss, the shape should be (num_samples,) and
      pred[i] is the probability of the positive class.
    sample_weight: a vector of weights of shape (num_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.
    from_logits: whether `pred` is logits or probability vector.
    small_value: a scalar. np.log can become -inf if the probability is too
      close to 0, so the probability is clipped below by small_value.

  Returns:
    the cross-entropy loss of each sample
  """
  if labels.shape[0] != pred.shape[0]:
    raise ValueError('labels and pred should have the same number of examples,',
                     f'but got {labels.shape[0]} and {pred.shape[0]}.')
  classes = np.unique(labels)
  if sample_weight is None:
    # If sample weights are not provided, set them to 1.0.
    sample_weight = 1.0
  else:
    if np.shape(sample_weight)[0] != np.shape(labels)[0]:
      # Number of elements should be the same.
      raise ValueError(
          'Expected sample weights to have the same length as the labels, '
          f'received {np.shape(sample_weight)[0]} and {np.shape(labels)[0]}.')

  # Binary logistic loss
  if pred.size == pred.shape[0]:
    pred = pred.flatten()
    if classes.min() < 0 or classes.max() > 1:
      raise ValueError('Each value in pred is a scalar, so labels are expected',
                       f'to be {0, 1}. But got {classes}.')
    if from_logits:
      pred = special.expit(pred)

    indices_class0 = (labels == 0)
    prob_correct = np.copy(pred)
    prob_correct[indices_class0] = 1 - prob_correct[indices_class0]
    return -np.log(np.maximum(prob_correct, small_value)) * sample_weight

  # Multi-class categorical cross entropy loss
  if classes.min() < 0 or classes.max() >= pred.shape[1]:
    raise ValueError('labels should be in the range [0, num_classes-1].')
  if from_logits:
    pred = special.softmax(pred, axis=-1)
  return (-np.log(np.maximum(pred[range(labels.size), labels], small_value)) *
          sample_weight)


def squared_loss(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
  """Computes the per-example squared loss.

  Args:
    y_true: numpy array of shape (num_samples,) representing the true labels.
    y_pred: numpy array of shape (num_samples,) representing the predictions.
    sample_weight: a vector of weights of shape (num_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.

  Returns:
    the squared loss of each sample.
  """
  if y_true.ndim != 1:
    logging.warning(('Squared loss expects the labels to have shape '
                     '(num_examples, ) but got shape %s. Will use np.squeeze.'),
                    y_true.shape)
    y_true = np.squeeze(y_true)
  if y_pred.ndim != 1:
    logging.warning(('Squared loss expects the predictions to have shape '
                     '(num_examples, ) but got shape %s. Will use np.squeeze.'),
                    y_pred.shape)
    y_pred = np.squeeze(y_pred)
  if y_true.shape != y_pred.shape:
    raise ValueError('Squared loss expects the labels and predictions to have '
                     'shape (num_examples, ), but after np.squeeze, the shapes '
                     'are %s and %s.' % (y_true.shape, y_pred.shape))
  if sample_weight is None:
    # If sample weights are not provided, set them to 1.0.
    sample_weight = 1.0
  return sample_weight * (y_true - y_pred)**2


def multilabel_bce_loss(labels: np.ndarray,
                        pred: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None,
                        from_logits=False,
                        small_value=1e-8) -> np.ndarray:
  """Computes the per-multi-label-example cross entropy loss.

  Args:
    labels: numpy array of shape (num_samples, num_classes). labels[i] is the
      true multi-hot encoded label (vector) of the i-th sample and each element
      of the vector is one of {0, 1}.
    pred: numpy array of shape (num_samples, num_classes).  pred[i] is the
      logits or probability vector of the i-th sample.
    sample_weight: a vector of weights of shape (num_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.
    from_logits: whether `pred` is logits or probability vector.
    small_value: a scalar. np.log can become -inf if the probability is too
      close to 0, so the probability is clipped below by small_value.

  Returns:
    the cross-entropy loss of each sample for each class.
  """
  # Check arrays.
  if labels.shape[0] != pred.shape[0]:
    raise ValueError('labels and pred should have the same number of examples,',
                     f'but got {labels.shape[0]} and {pred.shape[0]}.')
  if not ((labels == 0) | (labels == 1)).all():
    raise ValueError(
        'labels should be in {0, 1}. For multi-label classification the labels '
        'should be multihot encoded.')
  # Check if labels vectors are multi-label.
  summed_labels = np.sum(labels, axis=1)
  if ((summed_labels == 0) | (summed_labels == 1)).all():
    logging.info(
        ('Labels are one-hot encoded single label. Every sample has at most one'
         ' positive label.'))
  if not from_logits and ((pred < 0.0) | (pred > 1.0)).any():
    raise ValueError(('Prediction probabilities are not in [0, 1] and '
                      '`from_logits` is set to False.'))
  if sample_weight is None:
    # If sample weights are not provided, set them to 1.0.
    sample_weight = 1.0
  if isinstance(sample_weight, list):
    sample_weight = np.asarray(sample_weight)
  if isinstance(sample_weight, np.ndarray) and (sample_weight.ndim == 1):
    # NOMUTANTS--np.reshape(X, (-1, 1)) == np.reshape(X, (-N, 1)), N >=1.
    sample_weight = np.reshape(sample_weight, (-1, 1))

  # Multi-class multi-label binary cross entropy loss
  if from_logits:
    pred = special.expit(pred)
  bce = labels * np.log(pred + small_value)
  bce += (1 - labels) * np.log(1 - pred + small_value)
  return -bce * sample_weight


def string_to_loss_function(string: str):
  """Convert string to the corresponding LossFunction."""

  if string == LossFunction.CROSS_ENTROPY.value:
    return LossFunction.CROSS_ENTROPY
  if string == LossFunction.SQUARED.value:
    return LossFunction.SQUARED
  raise ValueError(f'{string} is not a valid loss function name.')


def get_loss(
    loss: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    logits: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    loss_function: LossFunctionCallable,
    loss_function_using_logits: Optional[bool],
    multilabel_data: Optional[bool],
    sample_weight: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
  """Calculates (if needed) losses.

  Args:
    loss: the loss of each example.
    labels: the scalar label of each example.
    logits: the logits vector of each example.
    probs: the probability vector of each example.
    loss_function: if `loss` is not available, `labels` and one of `logits`
      and `probs` are available, we will use this function to compute loss. It
      is supposed to take in (label, logits / probs) as input.
    loss_function_using_logits: if `loss_function` expects `logits` or
      `probs`.
    multilabel_data: if the data is from a multilabel classification problem.
    sample_weight: a vector of weights of shape (num_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.

  Returns:
    Loss (or None if neither the loss nor the labels are present).
  """
  if loss is not None:
    return loss
  if labels is None or (logits is None and probs is None):
    return None
  if loss_function_using_logits and logits is None:
    raise ValueError('We need logits to compute loss, but it is set to None.')
  if not loss_function_using_logits and probs is None:
    raise ValueError('We need probs to compute loss, but it is set to None.')

  predictions = logits if loss_function_using_logits else probs

  if isinstance(loss_function, str):
    loss_function = string_to_loss_function(loss_function)
  if loss_function == LossFunction.CROSS_ENTROPY:
    if multilabel_data:
      loss = multilabel_bce_loss(labels, predictions, sample_weight,
                                 loss_function_using_logits)
    else:
      loss = log_loss(labels, predictions, sample_weight,
                      loss_function_using_logits)
  elif loss_function == LossFunction.SQUARED:
    loss = squared_loss(labels, predictions, sample_weight)
  else:
    loss = loss_function(labels, predictions, sample_weight)
  return loss

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
"""Utility functions for membership inference attacks."""

from typing import Text, Dict, Union, List, Any, Tuple

import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults


ArrayDict = Dict[Text, np.ndarray]
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

# ------------------------------------------------------------------------------
#  Utilities for managing result dictionaries
# ------------------------------------------------------------------------------


def to_numpy(in_dict: Dict[Text, Any]) -> ArrayDict:
  """Convert values of dict to numpy arrays.

  Warning: This may fail if the values cannot be converted to numpy arrays.

  Args:
    in_dict: A dictionary mapping Text keys to values where the values must be
      something that can be converted to a numpy array.

  Returns:
    a dictionary with the same keys as input with all values converted to numpy
        arrays
  """
  return {k: np.array(v) for k, v in in_dict.items()}


def ensure_1d(in_dict: Dict[Text, Union[int, float, np.ndarray]]) -> ArrayDict:
  """Ensure all values of a dictionary are at least 1D numpy arrays.

  Args:
    in_dict: The input dictionary mapping Text keys to numpy arrays or numbers.

  Returns:
    dictionary with same keys as in_dict and values converted to numpy arrays
        with at least one dimension (i.e., pack scalars into arrays)
  """
  return {k: np.atleast_1d(v) for k, v in in_dict.items()}


def prepend_to_keys(in_dict: Dict[Text, Any], prefix: Text) -> Dict[Text, Any]:
  """Prepend a prefix to all keys of a dictionary.

  Args:
    in_dict: The input dictionary mapping Text keys to numpy arrays.
    prefix: Text which to prepend to each key in in_dict

  Returns:
    dictionary with same values as in_dict and all keys having prefix prepended
        to them
  """
  return {prefix + k: v for k, v in in_dict.items()}


# ------------------------------------------------------------------------------
#  Utilities for managing result.
# ------------------------------------------------------------------------------


def get_all_attack_results(results: AttackResults):
  """Get all results as a list of attack properties and a list of attack result."""
  properties = []
  values = []
  for attack_result in results.single_attack_results:
    slice_spec = attack_result.slice_spec
    prop = [str(slice_spec), str(attack_result.attack_type)]
    properties += [prop + ['adv'], prop + ['auc']]
    values += [float(attack_result.get_attacker_advantage()),
               float(attack_result.get_auc())]

  return properties, values


# ------------------------------------------------------------------------------
#  Subsampling and data selection functionality
# ------------------------------------------------------------------------------


def select_indices(in_dict: ArrayDict, indices: np.ndarray) -> ArrayDict:
  """Subsample all values in the dictionary by the provided indices.

  Args:
    in_dict: The input dictionary mapping Text keys to numpy array values.
    indices: A numpy which can be used to index other arrays, specifying the
      indices to subsample from in_dict values.

  Returns:
    dictionary with same keys as in_dict and subsampled values
  """
  return {k: v[indices] for k, v in in_dict.items()}


def merge_dictionaries(res: List[ArrayDict]) -> ArrayDict:
  """Convert iterable of dicts to dict of iterables."""
  output = {k: np.empty(0) for k in res[0]}
  for k in output:
    output[k] = np.concatenate([r[k] for r in res if k in r], axis=0)
  return output


def get_features(features: ArrayDict,
                 feature_name: Text,
                 top_k: int,
                 add_loss: bool = False) -> np.ndarray:
  """Combine the specified features into one array.

  Args:
    features: A dictionary containing all possible features.
    feature_name: Which feature to use (logits or prob).
    top_k: The number of the top features (of feature_name) to select.
    add_loss: Whether to also add the loss as a feature.

  Returns:
    combined numpy array with the selected features (n_examples, n_features)
  """
  if top_k < 1:
    raise ValueError('Must select at least one feature.')
  feats = np.sort(features[feature_name], axis=-1)[:, :top_k]
  if add_loss:
    feats = np.concatenate((feats, features['loss'][:, np.newaxis]), axis=-1)
  return feats


def subsample_to_balance(features: ArrayDict, random_state: int) -> ArrayDict:
  """Subsample if necessary to balance labels."""
  train_idx = features['is_train'] == 1
  test_idx = np.logical_not(train_idx)
  n0 = np.sum(test_idx)
  n1 = np.sum(train_idx)

  if n0 < 20 or n1 < 20:
    raise RuntimeError('Need at least 20 examples from training and test set.')

  np.random.seed(random_state)

  if n0 > n1:
    use_idx = np.random.choice(np.where(test_idx)[0], n1, replace=False)
    use_idx = np.concatenate((use_idx, np.where(train_idx)[0]))
    features = {k: v[use_idx] for k, v in features.items()}
  elif n0 < n1:
    use_idx = np.random.choice(np.where(train_idx)[0], n0, replace=False)
    use_idx = np.concatenate((use_idx, np.where(test_idx)[0]))
    features = {k: v[use_idx] for k, v in features.items()}

  return features


def get_train_test_split(features: ArrayDict, add_loss: bool,
                         test_size: float) -> Dataset:
  """Get training and test data split."""
  y = features['is_train']
  n_total = len(y)
  n_test = int(test_size * n_total)
  perm = np.random.permutation(len(y))
  test_idx = perm[:n_test]
  train_idx = perm[n_test:]
  y_train = y[train_idx]
  y_test = y[test_idx]

  # We are using 10 top logits as a good default value if there are more than 10
  # classes. Typically, there is no significant amount of weight in more than
  # 10 logits.
  n_logits = min(features['logits'].shape[1], 10)
  x = get_features(features, 'logits', n_logits, add_loss)

  x_train, x_test = x[train_idx], x[test_idx]
  return (x_train, y_train), (x_test, y_test)


# ------------------------------------------------------------------------------
#  Computation of the attack metrics
# ------------------------------------------------------------------------------


def compute_performance_metrics(true_labels: np.ndarray,
                                predictions: np.ndarray,
                                threshold: float = None) -> ArrayDict:
  """Compute relevant classification performance metrics.

  The outout metrics are
  1.arrays of thresholds and corresponding true and false positives (fpr, tpr).
  2.auc area under fpr-tpr curve.
  3.advantage max difference between tpr and fpr.
  4.precision/recall/accuracy/f1_score if threshold arg is given.

  Args:
    true_labels: True labels.
    predictions: Predicted probabilities/scores.
    threshold: The threshold to use on `predictions` binary classification.

  Returns:
    A dictionary with relevant metrics which are fully described by their key.
  """
  results = {}
  if threshold is not None:
    results.update({
        'precision':
            metrics.precision_score(true_labels, predictions > threshold),
        'recall':
            metrics.recall_score(true_labels, predictions > threshold),
        'accuracy':
            metrics.accuracy_score(true_labels, predictions > threshold),
        'f1_score':
            metrics.f1_score(true_labels, predictions > threshold),
    })

  fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
  auc = metrics.auc(fpr, tpr)
  advantage = np.max(np.abs(tpr - fpr))

  results.update({
      'fpr': fpr,
      'tpr': tpr,
      'thresholds': thresholds,
      'auc': auc,
      'advantage': advantage,
  })
  return ensure_1d(results)


# ------------------------------------------------------------------------------
#  Loss functions
# ------------------------------------------------------------------------------


def log_loss(y, pred, small_value=1e-8):
  """Compute the cross entropy loss.

  Args:
    y: numpy array, y[i] is the true label (scalar) of the i-th sample
    pred: numpy array, pred[i] is the probability vector of the i-th sample
    small_value: np.log can become -inf if the probability is too close to 0,
      so the probability is clipped below by small_value.

  Returns:
    the cross-entropy loss of each sample
  """
  return -np.log(np.maximum(pred[range(y.size), y], small_value))


# ------------------------------------------------------------------------------
#  Tensorboard
# ------------------------------------------------------------------------------


def write_to_tensorboard(writer, tags, values, step):
  """Write metrics to tensorboard.

  Args:
    writer: tensorboard writer
    tags: a list of tags of metrics
    values: a list of values of metrics
    step: step for the summary
  """
  if writer is None:
    return
  summary = tf.Summary()
  for tag, val in zip(tags, values):
    summary.value.add(tag=tag, simple_value=val)
  writer.add_summary(summary, step)
  writer.flush()

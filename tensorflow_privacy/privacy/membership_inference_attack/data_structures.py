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
"""Data structures representing attack inputs, configuration, outputs."""
import enum
import pickle
from typing import Any, Iterable, Union

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn import metrics


ENTIRE_DATASET_SLICE_STR = 'SingleSliceSpec(Entire dataset)'


class SlicingFeature(enum.Enum):
  """Enum with features by which slicing is available."""
  CLASS = 'class'
  PERCENTILE = 'percentile'
  CORRECTLY_CLASSIFIED = 'correctly_classfied'


@dataclass
class SingleSliceSpec:
  """Specifies a slice.

  The slice is defined by values in one feature - it might be a single value
  (eg. slice of examples of the specific classification class) or some set of
  values (eg. range of percentiles of the attacked model loss).

  When feature is None, it means that the slice is the entire dataset.
  """
  feature: SlicingFeature = None
  value: Any = None

  @property
  def entire_dataset(self):
    return self.feature is None

  def __str__(self):
    if self.entire_dataset:
      return 'Entire dataset'

    if self.feature == SlicingFeature.PERCENTILE:
      return 'Loss percentiles: %d-%d' % self.value

    return '%s=%s' % (self.feature.name, self.value)


@dataclass
class SlicingSpec:
  """Specification of a slicing procedure.

  Each variable which is set specifies a slicing by different dimension.
  """

  # When is set to true, one of the slices is the whole dataset.
  entire_dataset: bool = True

  # Used in classification tasks for slicing by classes. It is assumed that
  # classes are integers 0, 1, ... number of classes. When true one slice per
  # each class is generated.
  by_class: Union[bool, Iterable[int], int] = False

  # if true, it generates 10 slices for percentiles of the loss - 0-10%, 10-20%,
  # ... 90-100%.
  by_percentiles: bool = False

  # When true, a slice for correctly classifed and a slice for misclassifed
  # examples will be generated.
  by_classification_correctness: bool = False

  def __str__(self):
    """Only keeps the True values."""
    result = ['SlicingSpec(']
    if self.entire_dataset:
      result.append(' Entire dataset,')
    if self.by_class:
      if isinstance(self.by_class, Iterable):
        result.append(' Into classes %s,' % self.by_class)
      elif isinstance(self.by_class, int):
        result.append(' Up to class %d,' % self.by_class)
      else:
        result.append(' By classes,')
    if self.by_percentiles:
      result.append(' By percentiles,')
    if self.by_classification_correctness:
      result.append(' By classification correctness,')
    result.append(')')
    return '\n'.join(result)


class AttackType(enum.Enum):
  """An enum define attack types."""
  LOGISTIC_REGRESSION = 'lr'
  MULTI_LAYERED_PERCEPTRON = 'mlp'
  RANDOM_FOREST = 'rf'
  K_NEAREST_NEIGHBORS = 'knn'
  THRESHOLD_ATTACK = 'threshold'

  @property
  def is_trained_attack(self):
    """Returns whether this type of attack requires training a model."""
    return self != AttackType.THRESHOLD_ATTACK

  def __str__(self):
    """Returns LOGISTIC_REGRESSION instead of AttackType.LOGISTIC_REGRESSION."""
    return '%s' % self.name


def _is_integer_type_array(a):
  return np.issubdtype(a.dtype, np.integer)


def _is_last_dim_equal(arr1, arr1_name, arr2, arr2_name):
  """Checks whether the last dimension of the arrays is the same."""
  if arr1 is not None and arr2 is not None and arr1.shape[-1] != arr2.shape[-1]:
    raise ValueError('%s and %s should have the same number of features.' %
                     (arr1_name, arr2_name))


def _is_array_one_dimensional(arr, arr_name):
  """Checks whether the array is one dimensional."""
  if arr is not None and len(arr.shape) != 1:
    raise ValueError('%s should be a one dimensional numpy array.' % arr_name)


def _is_np_array(arr, arr_name):
  """Checks whether array is a numpy array."""
  if arr is not None and not isinstance(arr, np.ndarray):
    raise ValueError('%s should be a numpy array.' % arr_name)


@dataclass
class AttackInputData:
  """Input data for running an attack.

  This includes only the data, and not configuration.
  """

  logits_train: np.ndarray = None
  logits_test: np.ndarray = None

  # Contains ground-truth classes. Classes are assumed to be integers starting
  # from 0.
  labels_train: np.ndarray = None
  labels_test: np.ndarray = None

  # Explicitly specified loss. If provided, this is used instead of deriving
  # loss from logits and labels
  loss_train: np.ndarray = None
  loss_test: np.ndarray = None

  @property
  def num_classes(self):
    if self.labels_train is None or self.labels_test is None:
      raise ValueError(
          'Can\'t identify the number of classes as no labels were provided. '
          'Please set labels_train and labels_test')
    return int(max(np.max(self.labels_train), np.max(self.labels_test))) + 1

  @staticmethod
  def _get_loss(logits: np.ndarray, true_labels: np.ndarray):
    return logits[range(logits.shape[0]), true_labels]

  def get_loss_train(self):
    """Calculates cross-entropy losses for the training set."""
    if self.loss_train is not None:
      return self.loss_train
    return self._get_loss(self.logits_train, self.labels_train)

  def get_loss_test(self):
    """Calculates cross-entropy losses for the test set."""
    if self.loss_test is not None:
      return self.loss_test
    return self._get_loss(self.logits_test, self.labels_test)

  def get_train_size(self):
    """Returns size of the training set."""
    if self.loss_train is not None:
      return self.loss_train.size
    return self.logits_train.shape[0]

  def get_test_size(self):
    """Returns size of the test set."""
    if self.loss_test is not None:
      return self.loss_test.size
    return self.logits_test.shape[0]

  def validate(self):
    """Validates the inputs."""
    if (self.loss_train is None) != (self.loss_test is None):
      raise ValueError(
          'loss_test and loss_train should both be either set or unset')

    if (self.logits_train is None) != (self.logits_test is None):
      raise ValueError(
          'logits_train and logits_test should both be either set or unset')

    if (self.labels_train is None) != (self.labels_test is None):
      raise ValueError(
          'labels_train and labels_test should both be either set or unset')

    if (self.labels_train is None and self.loss_train is None and
        self.logits_train is None):
      raise ValueError('At least one of labels, logits or losses should be set')

    if self.labels_train is not None and not _is_integer_type_array(
        self.labels_train):
      raise ValueError('labels_train elements should have integer type')

    if self.labels_test is not None and not _is_integer_type_array(
        self.labels_test):
      raise ValueError('labels_test elements should have integer type')

    _is_np_array(self.logits_train, 'logits_train')
    _is_np_array(self.logits_test, 'logits_test')
    _is_np_array(self.labels_train, 'labels_train')
    _is_np_array(self.labels_test, 'labels_test')
    _is_np_array(self.loss_train, 'loss_train')
    _is_np_array(self.loss_test, 'loss_test')

    _is_last_dim_equal(self.logits_train, 'logits_train', self.logits_test,
                       'logits_test')
    _is_array_one_dimensional(self.loss_train, 'loss_train')
    _is_array_one_dimensional(self.loss_test, 'loss_test')
    _is_array_one_dimensional(self.labels_train, 'labels_train')
    _is_array_one_dimensional(self.labels_test, 'labels_test')

  def __str__(self):
    """Return the shapes of variables that are not None."""
    result = ['AttackInputData(']
    _append_array_shape(self.loss_train, 'loss_train', result)
    _append_array_shape(self.loss_test, 'loss_test', result)
    _append_array_shape(self.logits_train, 'logits_train', result)
    _append_array_shape(self.logits_test, 'logits_test', result)
    _append_array_shape(self.labels_train, 'labels_train', result)
    _append_array_shape(self.labels_test, 'labels_test', result)
    result.append(')')
    return '\n'.join(result)


def _append_array_shape(arr: np.array, arr_name: str, result):
  if arr is not None:
    result.append(' %s with shape: %s,' % (arr_name, arr.shape))


@dataclass
class RocCurve:
  """Represents ROC curve of a membership inference classifier."""
  # Thresholds used to define points on ROC curve.
  # Thresholds are not explicitly part of the curve, and are stored for
  # debugging purposes.
  thresholds: np.ndarray

  # True positive rates based on thresholds
  tpr: np.ndarray

  # False positive rates based on thresholds
  fpr: np.ndarray

  def get_auc(self):
    """Calculates area under curve (aka AUC)."""
    return metrics.auc(self.fpr, self.tpr)

  def get_attacker_advantage(self):
    """Calculates membership attacker's (or adversary's) advantage.

    This metric is inspired by https://arxiv.org/abs/1709.01604, specifically
    by Definition 4. The difference here is that we calculate maximum advantage
    over all available classifier thresholds.

    Returns:
      a single float number with membership attacker's advantage.
    """
    return max(np.abs(self.tpr - self.fpr))

  def __str__(self):
    """Returns AUC and advantage metrics."""
    return '\n'.join([
        'RocCurve(',
        '  AUC: %.2f' % self.get_auc(),
        '  Attacker advantage: %.2f' % self.get_attacker_advantage(), ')'
    ])


@dataclass
class SingleAttackResult:
  """Results from running a single attack."""

  # Data slice this result was calculated for.
  slice_spec: SingleSliceSpec

  attack_type: AttackType
  roc_curve: RocCurve  # for drawing and metrics calculation

  # TODO(b/162693190): Add more metrics. Think which info we should store
  #  to derive metrics like f1_score or accuracy. Should we store labels and
  #  predictions, or rather some aggregate data?

  def get_attacker_advantage(self):
    return self.roc_curve.get_attacker_advantage()

  def get_auc(self):
    return self.roc_curve.get_auc()

  def __str__(self):
    """Returns SliceSpec, AttackType, AUC and advantage metrics."""
    return '\n'.join([
        'SingleAttackResult(',
        '  SliceSpec: %s' % str(self.slice_spec),
        '  AttackType: %s' % str(self.attack_type),
        '  AUC: %.2f' % self.get_auc(),
        '  Attacker advantage: %.2f' % self.get_attacker_advantage(), ')'
    ])


@dataclass
class AttackResults:
  """Results from running multiple attacks."""
  # add metadata, such as parameters of attack evaluation, input data etc
  single_attack_results: Iterable[SingleAttackResult]

  def calculate_pd_dataframe(self):
    """Returns all metrics as a Pandas DataFrame."""
    slice_features = []
    slice_values = []
    attack_types = []
    advantages = []
    aucs = []

    for attack_result in self.single_attack_results:
      slice_spec = attack_result.slice_spec
      if slice_spec.entire_dataset:
        slice_feature, slice_value = 'entire_dataset', ''
      else:
        slice_feature, slice_value = slice_spec.feature.value, slice_spec.value
      slice_features.append(str(slice_feature))
      slice_values.append(str(slice_value))
      attack_types.append(str(attack_result.attack_type.value))
      advantages.append(float(attack_result.get_attacker_advantage()))
      aucs.append(float(attack_result.get_auc()))

    df = pd.DataFrame({'slice feature': slice_features,
                       'slice value': slice_values,
                       'attack type': attack_types,
                       'attack advantage': advantages,
                       'roc auc': aucs})
    return df

  def summary(self, by_slices=False) -> str:
    """Provides a summary of the metrics.

    The summary provides the best-performing attacks for each requested data
    slice.
    Args:
      by_slices : whether to prepare a per-slice summary.

    Returns:
      A string with a summary of all the metrics.
    """
    summary = []

    # Summary over all slices
    max_auc_result_all = self.get_result_with_max_attacker_advantage()
    summary.append('Best-performing attacks over all slices')
    summary.append(
        '  %s achieved an AUC of %.2f on slice %s' %
        (max_auc_result_all.attack_type, max_auc_result_all.get_auc(),
         max_auc_result_all.slice_spec))

    max_advantage_result_all = self.get_result_with_max_attacker_advantage()
    summary.append('  %s achieved an advantage of %.2f on slice %s' %
                   (max_advantage_result_all.attack_type,
                    max_advantage_result_all.get_attacker_advantage(),
                    max_advantage_result_all.slice_spec))

    slice_dict = self._group_results_by_slice()

    if len(slice_dict.keys()) > 1 and by_slices:
      for slice_str in slice_dict:
        results = slice_dict[slice_str]
        summary.append('\nBest-performing attacks over slice: \"%s\"' %
                       slice_str)
        max_auc_result = results.get_result_with_max_auc()
        summary.append('  %s achieved an AUC of %.2f' %
                       (max_auc_result.attack_type, max_auc_result.get_auc()))
        max_advantage_result = results.get_result_with_max_attacker_advantage()
        summary.append('  %s achieved an advantage of %.2f' %
                       (max_advantage_result.attack_type,
                        max_advantage_result.get_attacker_advantage()))

    return '\n'.join(summary)

  def _group_results_by_slice(self):
    """Groups AttackResults into a dictionary keyed by the slice."""
    slice_dict = {}
    for attack_result in self.single_attack_results:
      slice_str = str(attack_result.slice_spec)
      if slice_str not in slice_dict:
        slice_dict[slice_str] = AttackResults([])
      slice_dict[slice_str].single_attack_results.append(attack_result)
    return slice_dict

  def get_result_with_max_auc(self) -> SingleAttackResult:
    """Get the result with maximum AUC for all attacks and slices."""
    aucs = [result.get_auc() for result in self.single_attack_results]

    if min(aucs) < 0.4:
      print('Suspiciously low AUC detected: %.2f. ' +
            'There might be a bug in the classifier' % min(aucs))

    return self.single_attack_results[np.argmax(aucs)]

  def get_result_with_max_attacker_advantage(self) -> SingleAttackResult:
    """Get the result with maximum advantage for all attacks and slices."""
    return self.single_attack_results[np.argmax([
        result.get_attacker_advantage() for result in self.single_attack_results
    ])]

  def save(self, filepath):
    """Saves self to a pickle file."""
    with open(filepath, 'wb') as out:
      pickle.dump(self, out)

  @classmethod
  def load(cls, filepath):
    """Loads AttackResults from a pickle file."""
    with open(filepath, 'rb') as inp:
      return pickle.load(inp)

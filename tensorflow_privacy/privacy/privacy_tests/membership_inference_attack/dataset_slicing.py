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
"""Specifying and creating AttackInputData slices."""

from collections import abc
import copy
import logging
from typing import List, Optional

import numpy as np

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec


_MAX_NUM_OF_SLICES = 1000


def _slice_if_not_none(a, idx):
  return None if a is None else a[idx]


def _slice_data_by_indices(data: AttackInputData, idx_train,
                           idx_test) -> AttackInputData:
  """Slices train fields with idx_train and test fields with idx_test."""

  result = AttackInputData()

  # Slice train data.
  result.logits_train = _slice_if_not_none(data.logits_train, idx_train)
  result.probs_train = _slice_if_not_none(data.probs_train, idx_train)
  result.labels_train = _slice_if_not_none(data.labels_train, idx_train)
  result.loss_train = _slice_if_not_none(data.loss_train, idx_train)
  result.entropy_train = _slice_if_not_none(data.entropy_train, idx_train)
  # Slice sample weights if provided.
  result.sample_weight_train = _slice_if_not_none(data.sample_weight_train,
                                                  idx_train)

  # Slice test data.
  result.logits_test = _slice_if_not_none(data.logits_test, idx_test)
  result.probs_test = _slice_if_not_none(data.probs_test, idx_test)
  result.labels_test = _slice_if_not_none(data.labels_test, idx_test)
  result.loss_test = _slice_if_not_none(data.loss_test, idx_test)
  result.entropy_test = _slice_if_not_none(data.entropy_test, idx_test)
  # Slice sample weights if provided.
  result.sample_weight_test = _slice_if_not_none(data.sample_weight_test,
                                                 idx_test)

  # A slice has the same multilabel status as the original data. This is because
  # of the way multilabel status is computed. A dataset is multilabel if at
  # least 1 sample has a label that is multihot encoded with more than one
  # positive class. A slice of this dataset could have only samples with labels
  # that have only a single positive class, even if the original dataset were
  # multilable. Therefore we ensure that the slice inherits the multilabel state
  # of the original dataset.
  result.multilabel_data = data.is_multilabel_data()

  return result


def _slice_by_class(data: AttackInputData, class_value: int) -> AttackInputData:
  if data.is_multilabel_data():
    raise ValueError("Slicing by class not supported for multilabel data.")
  idx_train = data.labels_train == class_value
  idx_test = data.labels_test == class_value
  return _slice_data_by_indices(data, idx_train, idx_test)


def _slice_by_percentiles(data: AttackInputData, from_percentile: float,
                          to_percentile: float):
  """Slices samples by loss percentiles."""

  # Find from_percentile and to_percentile percentiles in losses.
  loss_train = data.get_loss_train()
  loss_test = data.get_loss_test()
  if data.is_multilabel_data():
    logging.info("For multilabel data, when slices by percentiles are "
                 "requested, losses are summed over the class axis before "
                 "slicing.")
    loss_train = np.sum(loss_train, axis=1)
    loss_test = np.sum(loss_test, axis=1)
  losses = np.concatenate((loss_train, loss_test))
  from_loss = np.percentile(losses, from_percentile)
  to_loss = np.percentile(losses, to_percentile)

  idx_train = (from_loss <= loss_train) & (loss_train <= to_loss)
  idx_test = (from_loss <= loss_test) & (loss_test <= to_loss)

  return _slice_data_by_indices(data, idx_train, idx_test)


def _indices_by_classification(logits_or_probs, labels, correctly_classified):
  idx_correct = labels == np.argmax(logits_or_probs, axis=1)
  return idx_correct if correctly_classified else np.invert(idx_correct)


def _slice_by_classification_correctness(data: AttackInputData,
                                         correctly_classified: bool):
  """Slices attack inputs by whether they were classified correctly.

  Args:
    data: Data to be used as input to the attack models.
    correctly_classified: Whether to use the indices corresponding to the
      correctly classified samples.

  Returns:
    AttackInputData object containing the sliced data.
  """

  if data.is_multilabel_data():
    raise ValueError("Slicing by classification correctness not supported for "
                     "multilabel data.")
  idx_train = _indices_by_classification(data.logits_or_probs_train,
                                         data.labels_train,
                                         correctly_classified)
  idx_test = _indices_by_classification(data.logits_or_probs_test,
                                        data.labels_test, correctly_classified)
  return _slice_data_by_indices(data, idx_train, idx_test)


def _slice_by_custom_indices(data: AttackInputData,
                             custom_train_indices: np.ndarray,
                             custom_test_indices: np.ndarray,
                             group_value: int) -> AttackInputData:
  """Slices attack inputs by custom indices.

  Args:
    data: Data to be used as input to the attack models.
    custom_train_indices: The group indices of each training example.
    custom_test_indices: The group indices of each test example.
    group_value: The group value to pick.

  Returns:
    AttackInputData object containing the sliced data.
  """
  train_size, test_size = data.get_train_size(), data.get_test_size()
  if custom_train_indices.shape[0] != train_size:
    raise ValueError(
        "custom_train_indices should have the same number of elements as "
        f"the training data, but got {custom_train_indices.shape} and "
        f"{train_size}")
  if custom_test_indices.shape[0] != test_size:
    raise ValueError(
        "custom_test_indices should have the same number of elements as "
        f"the test data, but got {custom_test_indices.shape} and "
        f"{test_size}")
  idx_train = custom_train_indices == group_value
  idx_test = custom_test_indices == group_value
  return _slice_data_by_indices(data, idx_train, idx_test)


def get_single_slice_specs(
    slicing_spec: SlicingSpec,
    num_classes: Optional[int] = None) -> List[SingleSliceSpec]:
  """Returns slices of data according to slicing_spec.

  Args:
    slicing_spec: the slicing specification
    num_classes: number of classes of the examples. Required when slicing by
      class.

  Returns:
    Slices of data according to the slicing specification.

  Raises:
    ValueError: If the number of slices is above `_MAX_NUM_OF_SLICES` when
      slicing by class or slicing with custom indices. Or, if `num_classes` is
      not provided when slicing by class.
  """
  result = []

  if slicing_spec.entire_dataset:
    result.append(SingleSliceSpec())

  # Create slices by class.
  by_class = slicing_spec.by_class
  if isinstance(by_class, bool):
    if by_class:
      if not num_classes:
        raise ValueError("When by_class == True, num_classes should be given.")
      if not 0 <= num_classes <= _MAX_NUM_OF_SLICES:
        raise ValueError(f"Too many classes for slicing by classes. "
                         f"Found {num_classes}."
                         f"Should be no more than {_MAX_NUM_OF_SLICES}.")
      for c in range(num_classes):
        result.append(SingleSliceSpec(SlicingFeature.CLASS, c))
  elif isinstance(by_class, int):
    result.append(SingleSliceSpec(SlicingFeature.CLASS, by_class))
  elif isinstance(by_class, abc.Iterable):
    for c in by_class:
      result.append(SingleSliceSpec(SlicingFeature.CLASS, c))

  # Create slices by percentiles
  if slicing_spec.by_percentiles:
    for percent in range(0, 100, 10):
      result.append(
          SingleSliceSpec(SlicingFeature.PERCENTILE, (percent, percent + 10)))

  # Create slices by correctness of the classifications.
  if slicing_spec.by_classification_correctness:
    result.append(SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, True))
    result.append(SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, False))

  # Create slices by custom indices.
  if slicing_spec.all_custom_train_indices:
    for custom_train_indices, custom_test_indices in zip(
        slicing_spec.all_custom_train_indices,
        slicing_spec.all_custom_test_indices):
      groups = np.intersect1d(
          np.unique(custom_train_indices),
          np.unique(custom_test_indices),
          assume_unique=True)
      if not 0 <= groups.size <= _MAX_NUM_OF_SLICES:
        raise ValueError(
            f"Too many groups ({groups.size}) for slicing by custom indices. "
            f"Should be no more than {_MAX_NUM_OF_SLICES}.")
      for g in groups:
        result.append(
            SingleSliceSpec(SlicingFeature.CUSTOM,
                            (custom_train_indices, custom_test_indices, g)))
  return result


def get_slice(data: AttackInputData,
              slice_spec: SingleSliceSpec) -> AttackInputData:
  """Returns a single slice of data according to slice_spec."""
  if slice_spec.entire_dataset:
    data_slice = copy.copy(data)
  elif slice_spec.feature == SlicingFeature.CLASS:
    data_slice = _slice_by_class(data, slice_spec.value)
  elif slice_spec.feature == SlicingFeature.PERCENTILE:
    from_percentile, to_percentile = slice_spec.value
    data_slice = _slice_by_percentiles(data, from_percentile, to_percentile)
  elif slice_spec.feature == SlicingFeature.CORRECTLY_CLASSIFIED:
    data_slice = _slice_by_classification_correctness(data, slice_spec.value)
  elif slice_spec.feature == SlicingFeature.CUSTOM:
    custom_train_indices, custom_test_indices, group_value = slice_spec.value
    data_slice = _slice_by_custom_indices(data, custom_train_indices,
                                          custom_test_indices, group_value)
  else:
    raise ValueError('Unknown slice spec feature "%s"' % slice_spec.feature)

  data_slice.slice_spec = slice_spec
  return data_slice

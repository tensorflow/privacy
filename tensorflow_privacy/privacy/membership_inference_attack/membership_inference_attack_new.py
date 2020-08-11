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
"""Code that runs membership inference attacks based on the model outputs.

This file belongs to the new API for membership inference attacks. This file
will be renamed to membership_inference_attack.py after the old API is removed.
"""

from typing import Iterable
import numpy as np
from sklearn import metrics

from tensorflow_privacy.privacy.membership_inference_attack import models
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.dataset_slicing import get_single_slice_specs
from tensorflow_privacy.privacy.membership_inference_attack.dataset_slicing import get_slice


def _get_slice_spec(data: AttackInputData) -> SingleSliceSpec:
  if hasattr(data, 'slice_spec'):
    return data.slice_spec
  return SingleSliceSpec()


def run_trained_attack(attack_input: AttackInputData, attack_type: AttackType):
  """Classification attack done by ML models."""
  attacker = None

  if attack_type == AttackType.LOGISTIC_REGRESSION:
    attacker = models.LogisticRegressionAttacker()
  elif attack_type == AttackType.MULTI_LAYERED_PERCEPTRON:
    attacker = models.MultilayerPerceptronAttacker()
  elif attack_type == AttackType.RANDOM_FOREST:
    attacker = models.RandomForestAttacker()
  elif attack_type == AttackType.K_NEAREST_NEIGHBORS:
    attacker = models.KNearestNeighborsAttacker()
  else:
    raise NotImplementedError('Attack type %s not implemented yet.' %
                              attack_type)

  prepared_attacker_data = models.create_attacker_data(attack_input)

  attacker.train_model(prepared_attacker_data.features_train,
                       prepared_attacker_data.is_training_labels_train)

  # Run the attacker on (permuted) test examples.
  predictions_test = attacker.predict(prepared_attacker_data.features_test)

  # Generate ROC curves with predictions.
  fpr, tpr, thresholds = metrics.roc_curve(
      prepared_attacker_data.is_training_labels_test, predictions_test)

  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  return SingleAttackResult(
      slice_spec=_get_slice_spec(attack_input),
      attack_type=attack_type,
      roc_curve=roc_curve)


def run_threshold_attack(attack_input: AttackInputData):
  fpr, tpr, thresholds = metrics.roc_curve(
      np.concatenate((np.zeros(attack_input.get_train_size()),
                      np.ones(attack_input.get_test_size()))),
      np.concatenate(
          (attack_input.get_loss_train(), attack_input.get_loss_test())))

  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  return SingleAttackResult(
      slice_spec=_get_slice_spec(attack_input),
      attack_type=AttackType.THRESHOLD_ATTACK,
      roc_curve=roc_curve)


def run_attack(attack_input: AttackInputData, attack_type: AttackType):
  attack_input.validate()
  if attack_type.is_trained_attack:
    return run_trained_attack(attack_input, attack_type)

  return run_threshold_attack(attack_input)


def run_attacks(
    attack_input: AttackInputData,
    slicing_spec: SlicingSpec = None,
    attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,)
) -> AttackResults:
  """Run all attacks."""
  attack_input.validate()
  attack_results = []

  if slicing_spec is None:
    slicing_spec = SlicingSpec(entire_dataset=True)
  input_slice_specs = get_single_slice_specs(slicing_spec,
                                             attack_input.num_classes)
  for single_slice_spec in input_slice_specs:
    attack_input_slice = get_slice(attack_input, single_slice_spec)
    for attack_type in attack_types:
      attack_results.append(run_attack(attack_input_slice, attack_type))

  return AttackResults(single_attack_results=attack_results)

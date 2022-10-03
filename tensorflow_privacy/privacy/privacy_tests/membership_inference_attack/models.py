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
"""Trained models for membership inference attacks."""

import contextlib
import dataclasses
import logging
from typing import Optional
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn.utils import parallel_backend

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import data_structures


@dataclasses.dataclass
class AttackerData:
  """Input data for an ML classifier attack.

  Labels in this class correspond to whether an example was in the
  train or test set.
  """
  # Features of in-training and out-of-training examples.
  features_all: Optional[np.ndarray] = None
  # Indicator for whether the example is in-training (0) or out-of-training (1).
  labels_all: Optional[np.ndarray] = None
  # Sample weights of in-training and out-of-training examples, if provided.
  sample_weights_all: Optional[np.ndarray] = None

  # Indices for `features_all` and `labels_all` that are going to be used for
  # training the attackers.
  fold_indices: Optional[np.ndarray] = None

  # Indices for `features_all` and `labels_all` that were left out due to
  # balancing. Disjoint from `fold_indices`.
  left_out_indices: Optional[np.ndarray] = None

  # Number of in-training and out-of-training examples.
  data_size: Optional[data_structures.DataSize] = None


def create_attacker_data(attack_input_data: data_structures.AttackInputData,
                         balance: bool = True) -> AttackerData:
  """Prepare AttackInputData to train ML attackers.

  Combines logits and losses and performs a random train-test split.

  Args:
    attack_input_data: Original AttackInputData
    balance: Whether the training and test sets for the membership inference
      attacker should have a balanced (roughly equal) number of samples from the
      training and test sets used to develop the model under attack.

  Returns:
    AttackerData.
  """
  attack_input_train = _column_stack(attack_input_data.logits_or_probs_train,
                                     attack_input_data.get_loss_train())
  attack_input_test = _column_stack(attack_input_data.logits_or_probs_test,
                                    attack_input_data.get_loss_test())

  ntrain, ntest = attack_input_train.shape[0], attack_input_test.shape[0]
  features_all = np.concatenate((attack_input_train, attack_input_test))
  labels_all = np.concatenate((np.zeros(ntrain), np.ones(ntest)))
  if attack_input_data.has_nonnull_sample_weights():
    sample_weights_all = np.concatenate((attack_input_data.sample_weight_train,
                                         attack_input_data.sample_weight_test),
                                        axis=0)
  else:
    sample_weights_all = None

  fold_indices = np.arange(ntrain + ntest)
  left_out_indices = np.asarray([], dtype=np.int32)

  if balance:
    idx_train, idx_test = range(ntrain), range(ntrain, ntrain + ntest)
    min_size = min(ntrain, ntest)
    if ntrain > min_size:
      left_out_size = ntrain - min_size
      perm_train = np.random.permutation(idx_train)  # shuffle training
      left_out_indices = perm_train[:left_out_size]
      fold_indices = np.concatenate((perm_train[left_out_size:], idx_test))
    elif ntest > min_size:
      left_out_size = ntest - min_size
      perm_test = np.random.permutation(idx_test)  # shuffle test
      left_out_indices = perm_test[:left_out_size]
      fold_indices = np.concatenate((perm_test[left_out_size:], idx_train))

  # Shuffle indices for the downstream attackers.
  fold_indices = np.random.permutation(fold_indices)

  return AttackerData(
      features_all=features_all,
      labels_all=labels_all,
      sample_weights_all=sample_weights_all,
      fold_indices=fold_indices,
      left_out_indices=left_out_indices,
      data_size=data_structures.DataSize(ntrain=ntrain, ntest=ntest))


def _sample_multidimensional_array(array, size):
  indices = np.random.choice(len(array), size, replace=False)
  return array[indices]


def _column_stack(logits, loss):
  """Stacks logits and losses.

  In case that only one exists, returns that one.
  Args:
    logits: logits array
    loss: loss array

  Returns:
    stacked logits and losses (or only one if both do not exist).
  """
  if logits is None:
    return np.expand_dims(loss, axis=-1)
  if loss is None:
    return logits
  return np.column_stack((logits, loss))


class TrainedAttacker(object):
  """Base class for training attack models.

  Attributes:
    backend: Name of Scikit-Learn parallel backend to use for this attack
      model. The default value of `None` performs single-threaded training.
    model: The trained attack model.
    ctx_mgr: The backend context manager within which to perform training.
      Defaults to the null context manager for single-threaded training.
    n_jobs: Number of jobs that can run in parallel when using a backend.
      Set to `1` for single-threading, and to `-1` for all parallel
      backends.
  """

  def __init__(self, backend: Optional[str] = None):
    self.model = None
    self.backend = backend
    if backend is None:
      # Default value of `None` will perform single-threaded training.
      self.ctx_mgr = contextlib.nullcontext()
      self.n_jobs = 1
      logging.info('Using single-threaded backend for training.')
    else:
      self.n_jobs = -1
      self.ctx_mgr = parallel_backend(
          # Values for 'backend': `loky`, `threading`, `multiprocessing`.
          # Can also use `dask`, `distributed`, `ray` if they are installed.
          backend=backend,
          n_jobs=self.n_jobs)
      logging.info('Using %s backend for training.', backend)

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    """Train an attacker model.

    This is trained on examples from train and test datasets.
    Args:
      input_features : array-like of shape (n_samples, n_features) Training
        vector, where n_samples is the number of samples and n_features is the
        number of features.
      is_training_labels : a vector of booleans of shape (n_samples, )
        representing whether the sample is in the training set or not.
      sample_weight: a vector of weights of shape (n_samples, ) that are
        assigned to individual samples. If not provided, then each sample is
        given unit weight. Only the LogisticRegressionAttacker and the
        RandomForestAttacker support sample weights.
    """
    raise NotImplementedError()

  def predict(self, input_features):
    """Predicts whether input_features belongs to train or test.

    Args:
      input_features : A vector of features with the same semantics as x_train
        passed to train_model.

    Returns:
      An array of probabilities denoting whether the example belongs to test.
    """
    if self.model is None:
      raise AssertionError(
          'Model not trained yet. Please call train_model first.')
    return self.model.predict_proba(input_features)[:, 1]


class LogisticRegressionAttacker(TrainedAttacker):
  """Logistic regression attacker."""

  def __init__(self, backend: Optional[str] = None):
    super().__init__(backend=backend)

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    with self.ctx_mgr:
      lr = linear_model.LogisticRegression(solver='lbfgs', n_jobs=self.n_jobs)
      param_grid = {
          'C': np.logspace(-4, 2, 10),
      }
      model = model_selection.GridSearchCV(
          lr, param_grid=param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
      model.fit(input_features, is_training_labels, sample_weight=sample_weight)
    self.model = model


class MultilayerPerceptronAttacker(TrainedAttacker):
  """Multilayer perceptron attacker."""

  def __init__(self, backend: Optional[str] = None):
    super().__init__(backend=backend)

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    del sample_weight  # MLP attacker does not use sample weights.
    with self.ctx_mgr:
      mlp_model = neural_network.MLPClassifier()
      param_grid = {
          'hidden_layer_sizes': [(64,), (32, 32)],
          'solver': ['adam'],
          'alpha': [0.0001, 0.001, 0.01],
      }
      model = model_selection.GridSearchCV(
          mlp_model, param_grid=param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
      model.fit(input_features, is_training_labels)
    self.model = model


class RandomForestAttacker(TrainedAttacker):
  """Random forest attacker."""

  def __init__(self, backend: Optional[str] = None):
    super().__init__(backend=backend)

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    """Setup a random forest pipeline with cross-validation."""
    with self.ctx_mgr:
      rf_model = ensemble.RandomForestClassifier(n_jobs=self.n_jobs)

      param_grid = {
          'n_estimators': [100],
          'max_features': ['auto', 'sqrt'],
          'max_depth': [5, 10, 20, None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
      }
      model = model_selection.GridSearchCV(
          rf_model, param_grid=param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
      model.fit(input_features, is_training_labels, sample_weight=sample_weight)
    self.model = model


class KNearestNeighborsAttacker(TrainedAttacker):
  """K nearest neighbor attacker."""

  def __init__(self, backend: Optional[str] = None):
    super().__init__(backend=backend)

  def train_model(self, input_features, is_training_labels, sample_weight=None):
    del sample_weight  # K-NN attacker does not use sample weights.
    with self.ctx_mgr:
      knn_model = neighbors.KNeighborsClassifier(n_jobs=self.n_jobs)
      param_grid = {
          'n_neighbors': [3, 5, 7],
      }
      model = model_selection.GridSearchCV(
          knn_model, param_grid=param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
      model.fit(input_features, is_training_labels)
    self.model = model


def create_attacker(attack_type,
                    backend: Optional[str] = None) -> TrainedAttacker:
  """Returns the corresponding attacker for the provided attack_type."""
  if attack_type == data_structures.AttackType.LOGISTIC_REGRESSION:
    return LogisticRegressionAttacker(backend=backend)
  if attack_type == data_structures.AttackType.MULTI_LAYERED_PERCEPTRON:
    return MultilayerPerceptronAttacker(backend=backend)
  if attack_type == data_structures.AttackType.RANDOM_FOREST:
    return RandomForestAttacker(backend=backend)
  if attack_type == data_structures.AttackType.K_NEAREST_NEIGHBORS:
    return KNearestNeighborsAttacker(backend=backend)
  raise NotImplementedError('Attack type %s not implemented yet.' % attack_type)

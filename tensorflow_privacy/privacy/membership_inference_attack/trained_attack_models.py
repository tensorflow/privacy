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
r"""A collection of sklearn models for binary classification.

This module contains some sklearn pipelines for finding models for binary
classification from a variable number of numerical input features.
These models are used to train binary classifiers for membership inference.
"""

from typing import Text

import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network


def choose_model(attack_classifier: Text):
  """Choose a model based on a string classifier."""
  if attack_classifier == 'lr':
    return logistic_regression()
  elif attack_classifier == 'mlp':
    return mlp()
  elif attack_classifier == 'rf':
    return random_forest()
  elif attack_classifier == 'knn':
    return knn()
  else:
    raise ValueError(f'Unknown attack classifier {attack_classifier}.')


def logistic_regression(verbose: int = 0, n_jobs: int = 1):
  """Setup a logistic regression pipeline with cross-validation."""
  lr = linear_model.LogisticRegression(solver='lbfgs')
  param_grid = {
      'C': np.logspace(-4, 2, 10),
  }
  pipe = model_selection.GridSearchCV(
      lr, param_grid=param_grid, cv=3, n_jobs=n_jobs, iid=False,
      verbose=verbose)
  return pipe


def random_forest(verbose: int = 0, n_jobs: int = 1):
  """Setup a random forest pipeline with cross-validation."""
  rf = ensemble.RandomForestClassifier()

  n_estimators = [100]
  max_features = ['auto', 'sqrt']
  max_depth = [5, 10, 20]
  max_depth.append(None)
  min_samples_split = [2, 5, 10]
  min_samples_leaf = [1, 2, 4]
  random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf}

  pipe = model_selection.RandomizedSearchCV(
      rf, param_distributions=random_grid, n_iter=7, cv=3, n_jobs=n_jobs,
      iid=False, verbose=verbose)
  return pipe


def mlp(verbose: int = 0, n_jobs: int = 1):
  """Setup a MLP pipeline with cross-validation."""
  mlpmodel = neural_network.MLPClassifier()

  param_grid = {
      'hidden_layer_sizes': [(64,), (32, 32)],
      'solver': ['adam'],
      'alpha': [0.0001, 0.001, 0.01],
  }
  pipe = model_selection.GridSearchCV(
      mlpmodel, param_grid=param_grid, cv=3, n_jobs=n_jobs, iid=False,
      verbose=verbose)
  return pipe


def knn(verbose: int = 0, n_jobs: int = 1):
  """Setup a k-nearest neighbors pipeline with cross-validation."""
  knnmodel = neighbors.KNeighborsClassifier()

  param_grid = {
      'n_neighbors': [3, 5, 7],
  }
  pipe = model_selection.GridSearchCV(
      knnmodel, param_grid=param_grid, cv=3, n_jobs=n_jobs, iid=False,
      verbose=verbose)
  return pipe

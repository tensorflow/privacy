# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Poisoning attack library for auditing."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def make_clip_aware(train_x, train_y, l2_norm=10):
  """
  train_x: clean training features - must be shape (n_samples, n_features)
  train_y: clean training labels - must be shape (n_samples, )

  Returns x, y1, y2
  x: poisoning sample
  y1: first corresponding y value
  y2: second corresponding y value
  """
  x_shape = list(train_x.shape[1:])
  to_image = lambda x: x.reshape([-1] + x_shape)  # reshapes to standard image shape
  flatten = lambda x: x.reshape((x.shape[0], -1))  # flattens all pixels - allows PCA

  # make sure to_image an flatten are inverse functions
  assert np.allclose(to_image(flatten(train_x)), train_x)

  flat_x = flatten(train_x)
  pca = PCA(flat_x.shape[1])
  pca.fit(flat_x)

  new_x = l2_norm*pca.components_[-1]

  lr = LogisticRegression(max_iter=1000)
  lr.fit(flat_x, np.argmax(train_y, axis=1))

  num_classes = train_y.shape[1]
  lr_probs = lr.predict_proba(new_x[None, :])
  min_y = np.argmin(lr_probs)
  second_y = np.argmin(lr_probs + np.eye(num_classes)[min_y])

  oh_min_y = np.eye(num_classes)[min_y]
  oh_second_y = np.eye(num_classes)[second_y]

  return to_image(new_x), oh_min_y, oh_second_y

def make_bkdr(train_x, train_y):
  """
  Makes a bkdred dataset, following Gu et al. https://arxiv.org/abs/1708.06733

  train_x: clean training features - must be shape (n_samples, n_features)
  train_y: clean training labels - must be shape (n_samples, )

  Returns x, y1, y2
  x: poisoning sample
  y1: first corresponding y value
  y2: second corresponding y value
  """

  sample_ind = np.random.choice(train_x.shape[0], 1)
  pois_x = np.copy(train_x[sample_ind, :])
  pois_x[0] = 1  # set corner feature to 1
  second_y = train_y[sample_ind]

  num_classes = train_y.shape[1]
  min_y = np.eye(num_classes)[second_y.argmax(1) + 1]

  return pois_x, min_y, second_y


def make_many_poisoned_datasets(train_x, train_y, pois_sizes, attack="clip_aware", l2_norm=10):
  """
  Makes a dict containing many poisoned datasets. make_pois is fairly slow:
  this avoids making multiple calls

  train_x: clean training features - shape (n_samples, n_features)
  train_y: clean training labels - shape (n_samples, )
  pois_sizes: list of poisoning sizes
  l2_norm: l2 norm of the poisoned data

  Returns dict: all_poisons
  all_poisons[poison_size] is a pair of poisoned datasets
  """
  if attack == "clip_aware":
    pois_sample_x, y, second_y = make_clip_aware(train_x, train_y, l2_norm)
  elif attack == "bkdr":
    pois_sample_x, y, second_y = make_bkdr(train_x, train_y)
  else:
    raise NotImplementedError
  all_poisons = {"pois": (pois_sample_x, y)}

  for pois_size in pois_sizes:  # make_pois is slow - don't want it in a loop
    new_pois_x1, new_pois_y1 = train_x.copy(), train_y.copy()
    new_pois_x2, new_pois_y2 = train_x.copy(), train_y.copy()

    new_pois_x1[-pois_size:] = pois_sample_x[None, :]
    new_pois_y1[-pois_size:] = y

    new_pois_x2[-pois_size:] = pois_sample_x[None, :]
    new_pois_y2[-pois_size:] = second_y

    dataset1, dataset2 = (new_pois_x1, new_pois_y1), (new_pois_x2, new_pois_y2)
    all_poisons[pois_size] = dataset1, dataset2

  return all_poisons

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""dataset loader"""

# pylint: skip-file
# pyformat: disable

import os
import ssl
import tarfile
import urllib.request
from my_logistic_regression import MyLogisticRegression
import numpy as np
import requests
from sklearn import preprocessing
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import torch
from torchvision import datasets, transforms


PATH_PREFIX = './src/datasets_directory'
ssl._create_default_https_context = ssl._create_unverified_context


def normalize_fvec(x_train):
  """normalize feature vectors"""
  feature_mean = np.mean(x_train, axis=0)
  feature_std = np.std(x_train, axis=0)
  x_train = (x_train - feature_mean) / feature_std
  return x_train


def backtracking_ls(lrp, dir_srch, w_start, alpha=0.4, beta=0.95):
  """Implementation of backtracking line search

  lr = logistic regression
  dir = the "noisy" gradient direction
  w_start = current point
  alpha and beta tradeoff the precision and complexity of the linesearch

  output is an (close to) optimal stepsize
  """
  step_size = 100
  val_0 = lrp.loss(w_start)
  inner_prod = np.dot(dir_srch, lrp.grad(w_start))
  while (
      lrp.loss(w_start - step_size * dir_srch)
      >= val_0 - step_size * alpha * inner_prod
  ):
    step_size = beta * step_size
    if step_size < 1e-6:
      break
  return step_size


def newton(dataset, w_init, bias=True):
  """Implementation of the newton method with linesearch without privacy constraints

  dataset = dataset
  w_init = initialization point

  output is the model parameter
  """
  feature_vecs, labels = dataset
  if bias is True:
    feature_vecs = np.hstack(
        (np.ones(shape=(np.shape(feature_vecs)[0], 1)), feature_vecs)
    )
  lrp = MyLogisticRegression(feature_vecs, labels, reg=1e-9)
  w_cur = w_init
  for _ in range(8):
    hess = lrp.hess(w_cur)
    dir_srch = np.linalg.solve(hess, lrp.grad_wor(w_cur))
    step_size = backtracking_ls(lrp, dir_srch, w_cur)
    w_cur = w_cur - step_size * dir_srch
  if lrp.loss_wor(w_cur) < lrp.loss_wor(w_init):
    w_out = w_cur
  else:
    w_out = w_init
  return w_out


class Mydatasets:
  """Represents datasets we use for expriments"""

  def __init__(self):
    data_dir = PATH_PREFIX + '/data'
    cache_dir = PATH_PREFIX + '/cache_datasets'
    if not os.path.exists(data_dir):
      os.mkdir(data_dir)
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)

  def find_optimal_classifier(self, dataset, bias=True):
    """find the optimal weight vector for the logistic regression

        for the problems with real datasets.

    dataset = training dataset
    bias = bias for the logistic model
    """
    inputs_vec, labels = dataset
    reg = 1e-9
    if bias is True:
      model_lr = LogisticRegression(max_iter=200, C=1 / reg).fit(
          inputs_vec, labels
      )
      w_opt1 = np.concatenate([model_lr.intercept_, np.squeeze(model_lr.coef_)])
      w_opt = newton(dataset, w_opt1, bias)
    else:
      model_lr = LogisticRegression(
          max_iter=200, fit_intercept=False, C=1 / reg
      ).fit(inputs_vec, labels)
      w_opt1 = np.squeeze(model_lr.coef_)
      w_opt = newton(dataset, w_opt1, bias)
    return w_opt

  def fmnist_dataset(self):
    """fmnist dataset"""
    transform_data = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    train_data_trans = datasets.FashionMNIST(
        root=PATH_PREFIX + '/data',
        download=True,
        train=True,
        transform=transform_data,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data_trans, batch_size=len(train_data_trans)
    )
    x_train = next(iter(train_loader))[0].numpy()
    x_train = x_train.reshape(len(x_train), -1)
    y_train = next(iter(train_loader))[1].numpy()
    label0 = 0
    label1 = 3
    indx0 = np.nonzero(y_train == label0)[0]
    indx1 = np.nonzero(y_train == label1)[0]
    labels = y_train.copy()
    labels[indx0] = -1
    labels[indx1] = 1
    indx = np.concatenate((indx0, indx1))
    x_train = x_train[indx]
    labels = labels[indx]
    dataset = x_train, labels
    w_opt = self.find_optimal_classifier(dataset, bias=False)
    return x_train, labels, w_opt

  def a1a_dataset(self):
    """a1a dataset"""
    a1a_url = (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t'
    )
    data_path = PATH_PREFIX + '/data/a1a'
    if not os.path.exists(data_path):
      _ = urllib.request.urlretrieve(a1a_url, data_path)
    data = sklearn.datasets.load_svmlight_file(data_path)
    inputs_vec, labels = data[0], data[1]
    inputs_vec = inputs_vec.toarray()
    scaler = preprocessing.StandardScaler().fit(inputs_vec)
    inputs_vec = scaler.transform(inputs_vec)
    labels = labels.astype(float)
    dataset = inputs_vec, labels
    w_opt = self.find_optimal_classifier(dataset)
    inputs_vec = np.hstack(
        (np.ones(shape=(np.shape(inputs_vec)[0], 1)), inputs_vec)
    )
    return inputs_vec, labels, w_opt

  def protein_dataset(self):
    """protein dataset"""
    path_protein = PATH_PREFIX + '/data/protein/'
    if not os.path.exists(path_protein):
      os.mkdir(path_protein)
      protein_url = (
          'https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz'
      )
      protein_file = PATH_PREFIX + '/data/protein/data_kddcup04.tar.gz'
      response = requests.get(protein_url, stream=True, timeout=100)
      if response.status_code == 200:
        with open(protein_file, 'wb') as file_data:
          file_data.write(response.raw.read())
      with tarfile.open(protein_file, 'r:gz') as tar:
        tar.extractall(path_protein)
    x_train = np.loadtxt(PATH_PREFIX + '/data/protein/bio_train.dat')[:, 3:]
    y_train = np.loadtxt(PATH_PREFIX + '/data/protein/bio_train.dat')[:, 2]
    indx0 = np.nonzero(y_train == 0)[0]
    indx1 = np.nonzero(y_train == 1)[0]
    labels = y_train.copy()
    labels[indx0] = -1
    labels[indx1] = 1
    indx = np.arange(len(x_train))
    np.random.seed(3000)
    indx_sample = np.random.choice(indx, 50000, replace=False)
    np.random.seed(None)
    x_train = x_train[indx_sample]
    labels = labels[indx_sample]
    x_train = normalize_fvec(x_train)
    w_opt = self.find_optimal_classifier((x_train, labels))
    x_train = np.hstack((np.ones(shape=(np.shape(x_train)[0], 1)), x_train))
    return x_train, labels, w_opt

  def synthetic_dataset(self, num_samples=10000, dim=100):
    """Generates a synthetic dataset for logistic regression.

    n = number of samples d = dimension Features are unit vectors (by default
    uniformly random). Labels are sampled from logistic distribution, so w is
    the "true" solution.
    """
    mean = np.zeros(dim)
    cov = np.eye(dim)
    inputs_vec_un = np.random.multivariate_normal(mean, cov, num_samples)
    nrm = np.linalg.norm(inputs_vec_un, axis=1)
    inputs_vec = inputs_vec_un * 1 / nrm[:, None]
    w_star = np.ones(dim)
    w_star[0] = 1
    inner_prod = np.dot(inputs_vec, w_star)
    params = np.exp(inner_prod) / (1 + np.exp(inner_prod))
    labels = 2 * np.random.binomial(1, params) - 1
    dataset = inputs_vec, labels
    w_opt = self.find_optimal_classifier(dataset, bias=False)
    return inputs_vec, labels, w_opt

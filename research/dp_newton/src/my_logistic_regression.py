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

"""logistic regression class and its methods"""

# pylint: skip-file
# pyformat: disable

import numpy as np


class MyLogisticRegression:
  """return a logistic regression problem

  There is a dataset consisting of features (vectors of norm <=1)
  and labels (+1,-1), represented as a numpy array.
  There is also an L2 regularizer.
  """

  def __init__(self, input_vecs, labels, reg=1e-8):
    """Initialize the data and the regularizer.

    X = n x d numpy array representing features
    y = n x 1 numpy array representing labels
    reg = L2 regularizing coefficient (to ensure solution is finite)

    Data will be rescaled so that ||X[i,:]|| * |y[i]| <= 1 for all i.
    """
    self.reg = float(reg)
    input_vecs = np.array(input_vecs)
    labels = np.array(labels)
    assert len(input_vecs.shape) == 2
    assert len(labels.shape) == 1
    self.input_vecs = input_vecs
    self.labels = labels
    self.num_samples, self.dim = input_vecs.shape
    assert labels.shape[0] == self.num_samples
    signed_data = input_vecs * labels[:, np.newaxis]
    norm = np.linalg.norm(signed_data, axis=1)
    scale = np.maximum(norm, np.ones_like(norm))
    self.data = (1 / scale[:, None]) * signed_data

  def loss(self, param):
    """Computes the loss represented by this object at w.

    If X,y is the data and reg is the regularizer, then the loss is (1/n)sum_i^n
    log(1+exp(-<w,X[i,:]*y[i]>)) + (reg/2)||w||^2
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, param))))
    reg_loss = 0.5 * self.reg * np.linalg.norm(param) ** 2
    return data_loss + reg_loss

  def loss_wor(self, param):
    """Computes the loss represented by this object at w without regularizer.

    If X,y is the data and reg is the regularizer, then the loss is
    (1/n)sum_i^n log(1+exp(-<w,X[i,:]*y[i]>))
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, param))))
    return data_loss

  def accuracy(self, param):
    """ " computes the accuracy of the model gievn by w"""
    score_pred = np.dot(self.input_vecs, param)
    label1_prob = np.where(
        score_pred >= 0,
        1 / (1 + np.exp(-score_pred)),
        np.exp(score_pred) / (1 + np.exp(score_pred)),
    )
    return np.mean(np.where(label1_prob >= 0.5, 1, -1) == self.labels)

  def grad(self, param, batch_idx=None):
    """Computes the gradient of the logistic regression at a given point w.

    If X,y is the data and reg is the regularizer, then the gradient is
    (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    if batch_idx is not None:
      data_batch = self.data[batch_idx]
    else:
      data_batch = self.data

    coeff_grad = -1 / (1 + np.exp(np.dot(data_batch, param)))
    data_grad = np.mean(data_batch * coeff_grad[:, np.newaxis], axis=0)
    return data_grad + self.reg * param

  def grad_wor(self, param, batch_idx=None):
    """Computes the gradient of the logistic regression at a given point w.

    If X,y is the data and reg is the regularizer, then the gradient is
    (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    if batch_idx is not None:
      data_batch = self.data[batch_idx]
    else:
      data_batch = self.data

    coeff_grad = -1 / (1 + np.exp(np.dot(data_batch, param)))
    data_grad = np.mean(data_batch * coeff_grad[:, np.newaxis], axis=0)
    return data_grad

  def hess(self, param, batch_idx=None):
    """Computes the Hessian of the logistic regression at a given point w.

    The Hessian is the matrix of second derivatives.

    If X,y is the data and reg is the regularizer, then the Hessian is
    (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
    where we assume y[i]^2==1.
    """
    if batch_idx is not None:
      data_batch = self.data[batch_idx]
      batch_size = len(batch_idx)
    else:
      data_batch = self.data
      batch_size = self.num_samples

    temp_var = np.dot(data_batch, param) / 2
    coeff_hess = 1 / (np.exp(temp_var) + np.exp(-temp_var)) ** 2
    raw_hess = np.dot(data_batch.T * coeff_hess, data_batch)
    return raw_hess / batch_size + self.reg * np.eye(self.dim)

  def hess_wor(self, param, batch_idx=None):
    """Computes the Hessian of the logistic regression at a given point w.

    The Hessian is the matrix of second derivatives.

    If X,y is the data, then the Hessian is
    (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
    where we assume y[i]^2==1.
    """
    if batch_idx is not None:
      data_batch = self.data[batch_idx]
      batch_size = len(batch_idx)
    else:
      data_batch = self.data
      batch_size = self.num_samples

    temp_var = np.dot(data_batch, param) / 2
    coeff_hess = 1 / (np.exp(temp_var) + np.exp(-temp_var)) ** 2
    raw_hess = np.dot(data_batch.T * coeff_hess, data_batch)
    return raw_hess / batch_size

  def upperbound(self, param, batch_idx=None):
    """Tightest universal quadratic upper bound on the loss function.

    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor This function gives
    the quadratic term (which replaces the Hessian)
    https://twitter.com/shortstein/status/1557961202256318464
    """

    if batch_idx is not None:
      data_batch = self.data[batch_idx]
      batch_size = len(batch_idx)
    else:
      data_batch = self.data
      batch_size = self.num_samples

    temp_var = -np.dot(data_batch, param)  # vector of y_i<x_i,w> for i in [n]
    # v = 0.5*np.tanh(a/2)/a, but, avoid 0/0 by special rule
    temp_var2 = np.divide(
        0.5 * np.tanh(temp_var / 2),
        temp_var,
        out=np.ones(temp_var.shape) * 0.25,
        where=np.abs(temp_var) > 1e-9,
    )
    hess_non = np.dot(data_batch.T * temp_var2, data_batch)
    return hess_non / batch_size + self.reg * np.eye(self.dim)

  def upperbound_wor(self, param, batch_idx=None):
    """Tightest universal quadratic upper bound on the loss function.

    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor This function gives
    the quadratic term (which replaces the Hessian)
    """
    if batch_idx is not None:
      data_batch = self.data[batch_idx]
      batch_size = len(batch_idx)
    else:
      data_batch = self.data
      batch_size = self.num_samples

    temp_var = -np.dot(data_batch, param)  # vector of y_i<x_i,w> for i in [n]
    temp_var2 = np.divide(
        0.5 * np.tanh(temp_var / 2),
        temp_var,
        out=np.ones(temp_var.shape) * 0.25,
        where=np.abs(temp_var) > 1e-9,
    )
    hess_non = np.dot(data_batch.T * temp_var2, data_batch)
    return hess_non / batch_size

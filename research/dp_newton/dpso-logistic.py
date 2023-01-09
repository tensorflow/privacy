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

# -*- coding: utf-8 -*-
"""Differentially Private Second-Order Methods for Logistic Regression.

This script implements several algorithms for DP logistic regression and
tests them on various datasets. It produces plots for our upcoming paper.

Code exported from Colab. Written by Mahdi Haghifam.

"""

# pylint: disable=invalid-name
#        We use upper case to denote matrices and lower case for vectors.
#        This conflicts with pylint's variable naming rules.
# pylint: disable=redefined-outer-name
#        This is a script hence we have "global" variables.
# pylint: disable=unused-argument
#        The update rule functions are meant to have the same signature,
#        so cannot just remove arguments. Ideally this should have been
#        implemented as a class, but much easier to define a function.

import math
import urllib.request

# from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
# from tensorflow import keras


class MyLogisticRegression:
  """Represents a logistic regression problem.

  There is a dataset consisting of features (vectors of norm <=1)
  and labels (+1,-1), represented as a numpy array.
  There is also an L2 regularizer.
  """

  def __init__(self, X, y, reg=1e-8):
    """Initialize the data and the regularizer.

    Args:
      X: n x d numpy array representing features
      y: n x 1 numpy array representing labels
      reg: L2 regularizing coefficient (to ensure solution is finite)

    Data will be rescaled so that ||X[i,:]|| * |y[i]| <= 1 for all i.
    """
    self.reg = float(reg)
    X = np.array(X)
    y = np.array(y)
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    self.n, self.d = X.shape
    assert y.shape[0] == self.n
    signed_data = X * y[:, np.newaxis]
    norm = np.linalg.norm(signed_data, axis=1)
    scale = np.maximum(norm, np.ones_like(norm))
    self.data = (1 / scale[:, None]) * signed_data

  def loss(self, w):
    """Computes the loss represented by this object at w.

    Args:
      w: weight vector

    Returns:
      If X,y is the data and reg is the regularizer, then the loss is
      (1/n)sum_i^n log(1+exp(-<w,X[i,:]*y[i]>)) + (reg/2)||w||^2
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, w))))
    reg_loss = 0.5 * self.reg * np.linalg.norm(w)**2
    return data_loss + reg_loss

  def loss_wor(self, w):
    """Computes the loss represented by this object at w without regularizer.

    Args:
      w: weight vector

    Returns:
      If X,y is the data and reg is the regularizer, then the loss is
      (1/n)sum_i^n log(1+exp(-<w,X[i,:]*y[i]>))
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, w))))
    return data_loss

  def grad(self, w):
    """Computes the gradient of the logistic regression at a given point w.

    Args:
      w: weight vector

    Returns:
      If X,y is the data and reg is the regularizer, then the gradient is
      (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    coeff_grad = -1/(1+np.exp(np.dot(self.data, w)))
    data_grad = np.mean(self.data * coeff_grad[:, np.newaxis], axis=0)
    return data_grad + self.reg * w

  def grad_wor(self, w):
    """Computes the gradient of the logistic regression at a given point w.

    Args:
      w: weight vector

    Returns:
      If X,y is the data and reg is the regularizer, then the gradient is
      (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    coeff_grad = -1/(1+np.exp(np.dot(self.data, w)))
    data_grad = np.mean(self.data * coeff_grad[:, np.newaxis], axis=0)
    return data_grad

  def hess(self, w):
    """Computes the Hessian of the logistic regression at a given point w.

    Args:
      w: weight vector

    Returns:
      The Hessian is the matrix of second derivatives.
      If X,y is the data and reg is the regularizer, then the Hessian is
      (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
      where we assume y[i]^2==1.
    """
    a = np.dot(self.data, w)/2
    coeff_hess = 1 / (np.exp(a)+np.exp(-a))**2
    raw_hess = np.dot(self.data.T * coeff_hess, self.data)
    return raw_hess/self.n + self.reg * np.eye(self.d)

  def hess_wor(self, w):
    """Computes the Hessian of the logistic regression at a given point w.

    Args:
      w: weight vector

    Returns:
      The Hessian is the matrix of second derivatives.
      If X,y is the data, then the Hessian is
      (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
      where we assume y[i]^2==1.
    """
    a = np.dot(self.data, w)/2
    coeff_hess = 1 / (np.exp(a)+np.exp(-a))**2
    raw_hess = np.dot(self.data.T * coeff_hess, self.data)
    return raw_hess/self.n

  def upperbound(self, w):
    """Computes tightest universal quadratic upper bound on the loss function.

    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor
    This function gives the quadratic term (which replaces the Hessian)
    https://twitter.com/shortstein/status/1557961202256318464

    Args:
      w: weight vector

    Returns:
      Matrix H such that for all v
      loss(v) <= loss(w)+<grad(w),w-v> + <H(w-v),w-v>/2
    """
    a = -np.dot(self.data, w)  # vector of y_i<x_i,w> for i in [n]
    # v = 0.5*np.tanh(a/2)/a
    # But avoid 0/0 by special rule
    v = np.divide(
        0.5 * np.tanh(a / 2),
        a,
        out=(np.ones(a.shape) * 0.25),
        where=(np.abs(a) > 1e-9))
    H = np.dot(self.data.T * v, self.data)
    return H / self.n + self.reg * np.eye(self.d)

  def upperbound_wor(self, w):
    """Computes tightest quadratic upper bound on the unregularized loss.

    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor
    This function gives the quadratic term (which replaces the Hessian)
    https://twitter.com/shortstein/status/1557961202256318464

    Args:
      w: weight vector

    Returns:
      Matrix H such that for all v
      loss(v) <= loss(w)+<grad(w),w-v> + <H(w-v),w-v>/2
    """
    a = -np.dot(self.data, w)  # vector of y_i<x_i,w> for i in [n]
    # v = 0.5*np.tanh(a/2)/a
    # But avoid 0/0 by special rule
    v = np.divide(
        0.5 * np.tanh(a / 2),
        a,
        out=(np.ones(a.shape) * 0.25),
        where=(np.abs(a) > 1e-9))
    H = np.dot(self.data.T * v, self.data)
    return H / self.n


class Mydatasets:
  """Represents datasets we use for testing the algorithms.
  """

  def __init__(self):
    pass

  def find_optimal_classifier(self, dataset, reg=1e-9):
    """Find the optimal weight vector for the logistic regression.

    Args:
      dataset: training dataset
      reg: regularizer

    Returns:
      Optimal weight vector.
    """
    X, y = dataset
    model_lr = LogisticRegression(max_iter=10000, C=1/reg).fit(X, y)
    w_opt1 = np.concatenate([model_lr.intercept_, np.squeeze(model_lr.coef_)])
    w_opt = newton(dataset, w_opt1)
    print("optimal weight vector norms", np.linalg.norm(w_opt))
    return w_opt

  def mnist_binary(self):
    """Download and extract MNIST data.

    We also select only two labels for the binary classification task.

    Returns:
      Features, labels, and optimal weight vector.
    """
    labels = [1, 7]
    label0, label1 = int(labels[0]), int(labels[1])
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    nrm = np.linalg.norm(x_train, axis=1)
    x_train = x_train * 1/nrm[:, None]
    y_train = y_train.astype(float)
    indx0 = np.nonzero(y_train == label0)[0]
    indx1 = np.nonzero(y_train == label1)[0]
    y_train[indx0] = -1
    y_train[indx1] = 1
    indx = np.concatenate((indx0, indx1))
    x_train = x_train[indx]
    labels = y_train[indx]
    dataset = x_train, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(x_train)[0], 1)),
                   x_train))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def w8a_dataset(self):
    """w8a dataset for logistic regression.
    """
    num_points = 15e3
    w8a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    data_path = "./w8a"
    urllib.request.urlretrieve(w8a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    selected_samples = np.random.choice(len(X), int(num_points))
    X = X[selected_samples, :]
    labels = labels[selected_samples]
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a1a_dataset(self):
    """Loads a1a dataset for logistic regression.
    """
    a1a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"
    data_path = "./a1a"
    urllib.request.urlretrieve(a1a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def phishing(self):
    """phishing dataset for logistic regression.
    """
    phishing_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"
    data_path = "./phishing"
    urllib.request.urlretrieve(phishing_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a5a_dataset(self):
    """a5a dataset for logistic regression.
    """
    a5a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a"
    data_path = "./a5a"
    urllib.request.urlretrieve(a5a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a6a_dataset(self):
    """a6a dataset for logistic regression.
    """
    a6a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a"
    data_path = "./a6a"
    urllib.request.urlretrieve(a6a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def madelon(self):
    """madelon dataset for logistic regression.
    """
    madelon_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon"
    data_path = "./madelon"
    urllib.request.urlretrieve(madelon_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def mushroom_dataset(self):
    """mushroom dataset for logistic regression.
    """
    mushroom_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
    data_path = "./mushrooms"
    urllib.request.urlretrieve(mushroom_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def synthetic_data(self, n=5000, d=50, cov=None, w=None):
    """Generates a synthetic dataset for logistic regression.

    Args:
      n: number of samples
      d: dimension
      cov: covariance of the data (optional, default: identity)
      w: true coefficient vector (optional, default:first standard basis vector)

    Returns:
      Synthetic dataset.
      Features are unit vectors (by default uniformly random).
      Labels are sampled from logistic distribution,
      where argument w is the "true" solution.
    """
    mean = np.zeros(d)
    if cov is None:
      cov = np.eye(d)
    X_un = np.random.multivariate_normal(mean, cov, n)
    nrm = np.linalg.norm(X_un, axis=1)
    X = X_un * 1/nrm[:, None]
    if w is None:
      w = np.ones(d)
      w[0] = 1
    inner_prod = np.dot(X, w)
    params = np.exp(inner_prod)/(1+np.exp(inner_prod))
    labels = 2*np.random.binomial(1, params)-1
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)),
                   X))  # adding a dummy dimension for the bias term.
    return X, labels, w_opt


class CompareAlgs:
  """Class to run multiple iterative algorithms and compare the results."""

  def __init__(self,
               lr,
               dataset,
               optimal_w,
               iters=10,
               w0=None,
               pb=None):
    """Initialize the problem."""
    X, _ = dataset
    self.w_opt = optimal_w
    n, d = np.shape(X)
    print("dataset is created: (number of samples, dimension)=" + str(n) + "," +
          str(d))

    if w0 is None:
      w0_un = np.random.multivariate_normal(np.zeros(d), np.eye(d))
      w0 = w0_un/np.linalg.norm(w0_un)
    self.w0 = w0  # initial iterate
    self.iters = iters
    self.pb = pb
    self.lr = lr
    self.plots = []  # List of lists of iterates
    self.names = []  # List of names
    self.linestyles = []  # List of line styles for plotting
    self.cutoff = 20 * np.linalg.norm(self.w_opt) + 20 * np.linalg.norm(
        self.w0) + 10  # how do you set this value? is it problem-specific?

  def add_plot(self, update_rule, name, linestyle):
    """Run an iterative update method & add to plot.

    Args:
      update_rule: a function that takes 4 arguments:
        current iterate
        LogisticRegression problem
        index of current iterate
        total number of iterations
        pb = privacy budget or similar
      name: string to display in legend
      linestyle: line style for plot
    """
    baseline = self.lr.loss_wor(self.w_opt)
    print(name)
    w = self.w0
    plot = [w]
    for i in range(self.iters):
      w = update_rule(w, self.lr, i, self.iters, self.pb)
      if np.linalg.norm(w) > self.cutoff:
        w = self.w0  # Stop things exploding
        print("Stop Things Exploding!")
      plot.append(w)
      print(
          str(i) + ": ||grad||=" + str(np.linalg.norm(self.lr.grad_wor(w))) +
          " ex_loss=" + str(self.lr.loss_wor(w) - baseline))
    self.plots.append(plot)
    self.names.append(name)
    self.linestyles.append(linestyle)
    print()

  def plot_grad_norms(self, legend=True):
    """Plot gradient norms for each iteration.
    """
    plt.clf()
    for plot, name, ls in zip(self.plots, self.names, self.linestyles):
      grad_norms = [np.linalg.norm(self.lr.grad_wor(w)) for w in plot]
      plt.plot(range(self.iters+1), grad_norms, ls, label=name)
    plt.yscale("log")
    ymax = np.linalg.norm(self.lr.grad(self.plots[0][0]))
    plt.ylim(top=ymax)
    # plt.ylim((1e-3, 1e-1))
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    if legend: plt.legend()
    plt.show()

  def loss_vals(self):
    """Outputs the loss vector for different methods.
    """
    baseline = self.lr.loss_wor(self.w_opt)
    loss_dict = {}
    for plot, name in zip(self.plots, self.names):
      losses = [self.lr.loss_wor(w)-baseline for w in plot]
      loss_dict[name] = [losses]
    return loss_dict

  def gradnorm_vals(self):
    """Outputs the gradient norm for different methods.
    """
    gradnorm_dict = {}
    for plot, name in zip(self.plots, self.names):
      grad_norms = [np.linalg.norm(self.lr.grad_wor(w)) for w in plot]
      gradnorm_dict[name] = [grad_norms]
    return gradnorm_dict

  def plot_losses(self):
    """Plots excess loss for each iteration.

        output is a dictionary where the keys are name of method and value is
        the loss for each iteration.
    """
    baseline = self.lr.loss_wor(self.w_opt)
    plt.clf()
    for plot, name, ls in zip(self.plots, self.names, self.linestyles):
      losses = [self.lr.loss_wor(w)-baseline for w in plot]
      plt.plot(range(self.iters+1), losses, ls, label=name)
    # plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.legend()
    plt.show()


def gd_priv(w, lr, i, iters, pb):
  """Implementation of DP-GD.

  Args:
    w: current point
    lr: logistic regression
    i: iteration number
    iters: total number of iterations
    pb: auxillary information

  Returns:
    The next iterate.
  """
  inv_lr_gd = 0.25  # We select the learning rate based on the smoothness
  sens = 1/(lr.n*(inv_lr_gd+lr.reg))  # Sensitivity
  rho = pb["total"] / iters  # divide total privacy budget up
  noise = np.random.normal(scale=sens/math.sqrt(2*rho), size=lr.d)
  return w - lr.grad(w)/(inv_lr_gd+lr.reg) + noise


def gd_priv_backtrackingls(w, lr, i, iters, pb):
  """Implementation of DP-GD with back-tracking line search.

  !!! this method is not private. We only use it as a baseline.

  Args:
    w: current point
    lr: logistic regression
    i: iteration number
    iters: total number of iterations
    pb: auxillary information

  Returns:
    The next iterate
  """
  rho_grad = pb["total"] / iters  # divide total privacy budget up
  grad_scale = (1/lr.n)*math.sqrt(0.5/rho_grad)
  grad_noise = np.random.normal(scale=grad_scale, size=lr.d)
  direction = lr.grad(w)+grad_noise
  stepsize_opt = backtracking_ls(lr, direction, w)
  return w - stepsize_opt * direction


def backtracking_ls(lr, direction, w_0, alpha=0.4, beta=0.95):
  """Implementation of backtracking line search.

  Args:
    lr: logistic regression
    direction: the "noisy" gradient direction
    w_0: current point
    alpha: tradeoff the precision and complexity of the linesearch
    beta: tradeoff the precision and complexity of the linesearch

  Returns:
    A good stepsize
  """
  t = 100
  while lr.loss(w_0 - t * direction
               ) >= lr.loss(w_0) - t * alpha * np.dot(direction, lr.grad(w_0)):
    t = beta * t
    if t < 1e-10:
      break
  return t


def newton(dataset, w_opt2):
  """Newton update rule.
  """
  X, y = dataset
  X = np.hstack((np.ones(shape=(np.shape(X)[0], 1)), X))
  lr = MyLogisticRegression(X, y, reg=1e-9)
  w_opt = w_opt2
  _, d = np.shape(X)
  w = np.zeros(d)
  for _ in range(30):
    H = lr.hess(w)
    direction = np.linalg.solve(H, lr.grad(w))
    step_size = backtracking_ls(lr, direction, w)
    w = w - step_size * direction
  if lr.loss_wor(w) < lr.loss_wor(w_opt2):
    w_opt = w
  return w_opt


def newton_ur(w, lr, i, iters, pb):
  H = lr.hess(w)
  direction = np.linalg.solve(H, lr.grad(w))
  step_size = backtracking_ls(lr, direction, w)
  return w - step_size * direction


class DoubleNoiseMech:
  """Our Method: Double Noise."""

  def __init__(self,
               lr,
               type_reg="add",
               hyper_tuning=False,
               curvature_info="hessian",
               plot_eigen=False):
    """Initializes the algorithm.

    Args:
      lr: logistic regression problem we are solving.
      type_reg: "add" or "clip" -- how we regularize eigenvalues.
      hyper_tuning: do we tune the hyperparameters.
      curvature_info: "hessian" or "ub" -- what quadratic we use.
      plot_eigen: show eigenvalues for debugging purposes.

    """
    self.type_reg = type_reg
    self.hyper_tuning = hyper_tuning
    self.curvature_info = curvature_info
    self.plot_eigen = plot_eigen
    if self.curvature_info == "hessian":
      self.H = lr.hess_wor
    elif self.curvature_info == "ub":
      self.H = lr.upperbound_wor

  def find_opt_reg_wop(self, w, lr, noisy_grad, rho_hess):
    """Implementation of finding the optimal lambda.

     Here, we don't pay for privacy of doing it.

    Args:
      w: current point
      lr: logistic regression problem
      noisy_grad: the gradient estimate
      rho_hess: the privacy budget

    Returns:
      The next iterate.
    """
    increase_factor = 1.5  # at each step we increase the clipping
    if self.type_reg == "add":
      lambda_cur = 5e-6  # starting parameter
    elif self.type_reg == "clip":
      lambda_cur = 0.25/lr.n + 1e-5  # starting parameter,
    num_noise_sample = 5  # we want to estimate expected value over the noise
    grad_norm = np.linalg.norm(noisy_grad)
    H = self.H(w)
    best_loss = 1e6  # a large dummy number
    while lambda_cur <= 0.25:
      H = self.hess_mod(w, lambda_cur)
      if self.type_reg == "add":  # Sensitivity is different for add vs clip
        sens2 = grad_norm * 0.25/(lr.n*lambda_cur**2 + 0.25*lambda_cur)
      elif self.type_reg == "clip":
        sens2 = grad_norm * 0.25/(lr.n*lambda_cur**2 - 0.25*lambda_cur)
      loss_ = 0
      for _ in range(num_noise_sample):
        noise2 = np.random.normal(scale=sens2*math.sqrt(.5/rho_hess), size=lr.d)
        loss_ = loss_ + lr.loss_wor(w - np.linalg.solve(H, noisy_grad) + noise2)
      if loss_ < best_loss:
        best_loss = loss_
        lambda_star = lambda_cur
      lambda_cur = lambda_cur * increase_factor
    return lambda_star

  def update_rule(self, w, lr, i, iters, pb):
    """update rule."""
    total = pb["total"]
    grad_frac = pb["grad_frac4"]
    rho1 = grad_frac * total / iters  # divide total privacy budget for gradient
    rho2 = (1-grad_frac) * total / iters  # divide total privacy budget
    sc1 = (1/lr.n) * math.sqrt(0.5/rho1)
    noise1 = np.random.normal(scale=sc1, size=lr.d)
    noisy_grad = lr.grad(w)+noise1
    grad_norm = np.linalg.norm(noisy_grad)
    m = 0.25  # smoothness parameter
    frac_trace = 0.1  # fraction of privacy budget for estimating the trace.
    H = self.H(w)
    if self.plot_eigen:
      val, _ = np.linalg.eigh(H)
      hist, bin_edges = np.histogram(val, bins=300, range=(0, 0.01))
      cdf_vals = np.cumsum(hist)
      plt.clf()
      plt.plot(bin_edges[1:], cdf_vals)
      plt.show()
    if self.hyper_tuning:
      min_eval = self.find_opt_reg_wop(w, lr, noisy_grad, rho2)
      print("optimized min_eval", min_eval)
    else:
      noisy_trace = max(
          np.trace(H) + np.random.normal(
              scale=(0.25 / lr.n) * math.sqrt(0.5 / (frac_trace * rho2))), 0)
      min_eval = (noisy_trace / ((lr.n)**2 *
                                 (1 - frac_trace) * rho2))**(1 / 3) + 5e-4
      print("approx min_eval ", min_eval)

    H = self.hess_mod(w, min_eval, lr.reg)
    if self.type_reg == "add":  # Sensitivity is different for add vs clip
      sens2 = grad_norm * m/(lr.n * min_eval**2 + m * min_eval)
    elif self.type_reg == "clip":
      sens2 = grad_norm * m / (lr.n * min_eval**2 - m * min_eval)
    noise2 = np.random.normal(
        scale=sens2 * math.sqrt(0.5 / ((1 - frac_trace) * rho2)), size=lr.d)
    return w - np.linalg.solve(H, noisy_grad) + noise2

  def hess_mod(self, w, min_eval, reg=1e-9):
    if self.type_reg == "clip":
      evals, evec = np.linalg.eigh(self.H(w))
      # true_min = np.min(evals)
      evals = np.maximum(evals, min_eval*np.ones(evals.shape))
      Hclipped = np.dot(evec * evals, evec.T)
      return Hclipped
    elif self.type_reg == "add":
      return  self.H(w) + min_eval*np.eye(len(self.H(w)))


def helper_fun(datasetname, pb, num_rep=5, Tuning=False, plot_eigen=False):
  """This function loads the data & runs the algorithms.

  Args:
    datasetname: name of the dataset
    pb: a dictionary with the parameters
    num_rep: number of times we repeat the optimization for reporting average
    Tuning: True or False exhustive search for fining the best min eigenval
    plot_eigen: Show eigenvalues

  Returns:
    losses and gradient norms
  """
  datasets = Mydatasets()
  X, y, w_opt = getattr(datasets, datasetname)()
  dataset = X, y
  lr = MyLogisticRegression(X, y, reg=1e-8)
  dnm_hess_add = DoubleNoiseMech(
      lr,
      type_reg="add",
      hyper_tuning=False,
      curvature_info="hessian",
      plot_eigen=plot_eigen).update_rule
  dnm_ub_add = DoubleNoiseMech(
      lr,
      type_reg="add",
      hyper_tuning=False,
      curvature_info="ub",
      plot_eigen=plot_eigen).update_rule
  dnm_hess_clip = DoubleNoiseMech(
      lr,
      type_reg="clip",
      hyper_tuning=False,
      curvature_info="hessian",
      plot_eigen=plot_eigen).update_rule
  dnm_ub_clip = DoubleNoiseMech(
      lr,
      type_reg="clip",
      hyper_tuning=False,
      curvature_info="ub",
      plot_eigen=plot_eigen).update_rule
  if Tuning:
    # dnm_hess_add_ht = DoubleNoiseMech(lr,type_reg='add',
    #     hyper_tuning=True,curvature_info='hessian').update_rule
    # dnm_ub_add_ht = DoubleNoiseMech(lr,type_reg='add',
    #     hyper_tuning=True,curvature_info='ub').update_rule
    dnm_hess_clip_ht = DoubleNoiseMech(
        lr,
        type_reg="clip",
        hyper_tuning=True,
        curvature_info="hessian",
        plot_eigen=plot_eigen).update_rule
    # dnm_ub_clip_ht = DoubleNoiseMech(lr,type_reg='clip',
    #     hyper_tuning=True,curvature_info='ub').update_rule
  c = CompareAlgs(lr, dataset, w_opt, iters=10, pb=pb)
  for rep in range(num_rep):
    c.add_plot(gd_priv, "DPGD", "y--")
    c.add_plot(dnm_hess_add, "DN-Hess-add", "k-")
    c.add_plot(dnm_ub_add, "DN-UB-add", "b-")
    c.add_plot(dnm_hess_clip, "DN-Hess-clip", "k*-")
    c.add_plot(dnm_ub_clip, "DN-UB-clip", "b*-")
    c.add_plot(gd_priv_backtrackingls, "DP-GD-Oracle", "m")
    if Tuning:
      c.add_plot(dnm_hess_clip_ht, "DN-Hess-clip-T", "r*-")
      # c.add_plot(dnm_hess_add_ht,"DN-Hess-add-T",'r-')
      # c.add_plot(dnm_ub_clip_ht,"DN-UB-clip-T",'g*-')
      # c.add_plot(dnm_ub_add_ht,"DN-UB-add-T",'g-')
    losses_dict = c.loss_vals()
    gradnorm_dict = c.gradnorm_vals()
    if rep == 0:
      losses_total = losses_dict
      gradnorm_total = gradnorm_dict
    else:
      for names in losses_total:
        losses_total[names].extend(losses_dict[names])
        gradnorm_total[names].extend(gradnorm_dict[names])
  return losses_total, gradnorm_total

linestyle = {
    "DPGD": "y-",
    "DN-Hess-add": "k+-",
    "DN-UB-add": "b-",
    "DN-Hess-clip": "r*-",
    "DN-UB-clip": "g-",
    "DP-GD-Oracle": "c-"
}
facecolor = {
    "DPGD": "yellow",
    "DN-Hess-add": "black",
    "DN-UB-add": "blue",
    "DN-Hess-clip": "red",
    "DN-UB-clip": "green",
    "DP-GD-Oracle": "cyan"
}
alg_plt = [
    "DPGD",
    "DN-Hess-add",
    "DN-UB-add",
    "DN-Hess-clip",
    "DN-UB-clip",
    "DP-GD-Oracle"
]

# Synthethic Data

pb = {
    "total": 1,  # Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4":
        0.75  # Fraction of privacy budget for gradient vs matrix sensitivity
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "synthetic_data", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(1)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("synth.pdf")
    plt.figure(2)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("synth-grad.pdf")

# a5a dataset

pb = {
    "total": 1,  # Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4":
        0.75  # Fraction of privacy budget for gradient vs matrix sensitivity
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "a5a_dataset", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(3)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("a5a.pdf")
    plt.figure(4)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("a5a-grad.pdf")

# w8a dataset

pb = {
    "total": 1,  # Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4":
        0.75  # Fraction of privacy budget for gradient vs matrix sensitivity
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "w8a_dataset", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(5)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("w8a.pdf")
    plt.figure(6)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("w8a-grad.pdf")

# a1a dataset

pb = {
    "total": 1,  #  Total privacy budget for zCDP
    "min_eval4": 5e-3,  #  Min eigenvalue for clipping
    "grad_frac4": 0.75  #  Fraction of privacy budget for gradient vs matrix
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "a1a_dataset", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(7)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("a1a.pdf")
    plt.figure(8)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("a1a-grad.pdf")

# mushroom dataset

pb = {
    "total": 1,  #  Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4": 0.75  #  Fraction of privacy budget for gradient vs matrix
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "mushroom_dataset", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(9)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("mushroom.pdf")
    plt.figure(10)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("mushroom-grad.pdf")

# MNIST

pb = {
    "total": 1,  #  Total privacy budget for zCDP
    "min_eval4": 5e-3,  #  Min eigenvalue for clipping
    "grad_frac4": 0.75  #  Fraction of privacy budget for gradient vs matrix
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "mnist_binary", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(11)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("mnist.pdf")
    plt.figure(12)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("mnist-grad.pdf")

# Dataset: phishing

pb = {
    "total": 1,  # Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4": 0.75  # Fraction of privacy budget for gradient vs matrix
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "phishing", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(13)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("phishing.pdf")
    plt.figure(14)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("phishing-grad.pdf")

# Dataset: Madelon

# pb = {
#     "total": 1,  # Total privacy budget for zCDP
#     "min_eval4":  5e-3,  # Min eigenvalue for clipping
#     "grad_frac4": 0.4 # Fraction of privacy budget for gradient vs matrix
# }
# num_rep = 1
# losses_total,gradnorm_total = helper_fun('madelon',pb,num_rep = num_rep,
#     Tuning=True,plot_eigen=True)
# for alg in losses_total.keys():
#     losses = np.array(losses_total[alg])
#     gradnorm = np.array(gradnorm_total[alg])
#     loss_avg, gradnorm_avg = np.mean(losses,axis=0), np.mean(gradnorm,axis=0)
#     loss_std, gradnorm_std  = np.std(losses,axis=0)/np.sqrt(num_rep),
#         np.std(gradnorm,axis=0)/np.sqrt(num_rep)
#     print(str(alg)+ ':' + " ex_loss="+str(loss_avg[-1])+ ',
#         std='+str(loss_std[-1]))
#     if alg in alg_plt:
#         iters = len(loss_avg)
#         plt.figure(1)
#         plt.plot(range(iters),loss_avg,linestyle[alg],label=alg)
#         plt.fill_between(range(iters), loss_avg-loss_std, loss_avg+loss_std,
#             facecolor=facecolor[alg])
#         plt.legend()
#         plt.xlabel("Iteration")
#         plt.ylabel("Excess Loss")
#         plt.savefig('madelon.pdf')
#         plt.figure(2)
#         plt.plot(range(iters),gradnorm_avg,linestyle[alg],label=alg)
#         plt.yscale('log')
#         plt.legend()
#         plt.xlabel("Iteration")
#         plt.ylabel("Norm of Gradient")
#         plt.savefig('madelon-grad.pdf')

# Test) a6a Dataset

pb = {
    "total": 1,  # Total privacy budget for zCDP
    "min_eval4": 5e-3,  # Min eigenvalue for clipping
    "grad_frac4": 0.75  # Fraction of privacy budget for gradient vs matrix
}
num_rep = 30
losses_total, gradnorm_total = helper_fun(
    "a6a_dataset", pb, num_rep=num_rep, Tuning=False)
for alg in losses_total:
  losses = np.array(losses_total[alg])
  gradnorm = np.array(gradnorm_total[alg])
  loss_avg, gradnorm_avg = np.mean(losses, axis=0), np.mean(gradnorm, axis=0)
  loss_std, gradnorm_std = np.std(
      losses, axis=0) / np.sqrt(num_rep), np.std(
          gradnorm, axis=0) / np.sqrt(num_rep)
  print(
      str(alg) + ":" + " ex_loss=" + str(loss_avg[-1]) + ", std=" +
      str(loss_std[-1]))
  if alg in alg_plt:
    iters = len(loss_avg)
    plt.figure(15)
    plt.plot(range(iters), loss_avg, linestyle[alg], label=alg)
    plt.fill_between(
        range(iters),
        loss_avg - loss_std,
        loss_avg + loss_std,
        facecolor=facecolor[alg])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Excess Loss")
    plt.savefig("a6a.pdf")
    plt.figure(16)
    plt.plot(range(iters), gradnorm_avg, linestyle[alg], label=alg)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.savefig("a6a-grad.pdf")

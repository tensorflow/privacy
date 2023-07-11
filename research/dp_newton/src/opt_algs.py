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

"""file containing all auxillary functions for running the optimization algorithms"""

# pylint: skip-file
# pyformat: disable

import time
from my_logistic_regression import MyLogisticRegression
import numpy as np


class CompareAlgs:
  """Class to run multiple iterative algorithms and compare the results."""

  def __init__(self, lrp, optimal_w, hyper_dict):
    """Initialize the problem.

    lr = an instance of MyLogisticRegression
    dataset = dataset in the format of (features,label)
    optimal_w = optimal minimizer of logistic loss on dataset without privacy
    pb = hyperparameters
    """
    self.w_opt = optimal_w
    self.lrp = lrp
    self.iters = hyper_dict["num_iteration"]
    self.hyper_params = hyper_dict
    self.clock_time = []
    self.params = []
    self.names = []

  def add_algo(self, update_rule, name):
    """Run an iterative update method"""
    _, dim = self.lrp.num_samples, self.lrp.dim
    wint_un = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
    w_int = wint_un / np.linalg.norm(wint_un)
    cutoff_norm = (
        100 * np.linalg.norm(self.w_opt) + 100 * np.linalg.norm(w_int) + 100
    )
    w_cur = w_int
    params = [w_cur]
    start_t = time.time()
    wall_clock = [0]
    for _ in range(self.iters):
      w_cur = update_rule(w_cur, self.lrp, self.hyper_params)
      if np.linalg.norm(w_cur) > cutoff_norm:
        w_cur = w_int
        print("Stop Things Exploding!")
      params.append(w_cur)
      wall_clock.append(time.time() - start_t)
    self.clock_time.append(wall_clock)
    self.params.append(params)
    self.names.append(name)

  def wall_clock_alg(self):
    """compute the wall clock of different algorithms"""
    clock_time_dict = {}
    for time_alg, name in zip(self.clock_time, self.names):
      clock_time_dict[name] = [time_alg]
    return clock_time_dict

  def loss_vals(self):
    """output the loss per iteration for different methods"""
    baseline = self.lrp.loss_wor(self.w_opt)
    loss_dict = {}
    for params, name in zip(self.params, self.names):
      losses = [self.lrp.loss_wor(w) - baseline for w in params]
      loss_dict[name] = [losses]
    return loss_dict

  def accuracy_vals(self):
    """output the accuracy per iteration for different methods"""
    acc_dict = {}
    for params, name in zip(self.params, self.names):
      acc_vec = [self.lrp.accuracy(w) for w in params]
      acc_dict[name] = [acc_vec]
    return acc_dict

  def accuracy_np(self):
    """output the accuracy of the optimal model without privacy"""
    return self.lrp.accuracy(self.w_opt)

  def gradnorm_vals(self):
    """output the gradient norm per iteration for different methods"""
    gradnorm_dict = {}
    for params, name in zip(self.params, self.names):
      grad_norms = [np.linalg.norm(self.lrp.grad_wor(w)) for w in params]
      gradnorm_dict[name] = [grad_norms]
    return gradnorm_dict


def private_newton(w_cur, lrp, hyper_dict):
  """implementation of private newton method from [ABL21]

  w = current iterate
  lr = an instance of MyLogisticRegression
  i = the index of current iterate
  iters = total number of iterations
  pb =  privacy budget and other info
  return the next iterate
  """
  total = hyper_dict["total"]
  grad_frac = hyper_dict["grad_frac"]
  iters = hyper_dict["num_iteration"]
  hess = lrp.hess(w_cur)
  rho_grad = grad_frac * total / iters  # divide total privacy budget up.
  rho_hess = (1 - grad_frac) * total / iters
  hess_noise = np.random.normal(
      scale=(0.25 / lrp.num_samples) * np.sqrt(0.5 / rho_hess),
      size=(lrp.dim, lrp.dim),
  )
  hess_noise = (hess_noise + hess_noise.T) / 2
  hess_noisy = eigenclip(hess + hess_noise)
  grad = lrp.grad(w_cur)
  grad_noisy = grad + np.random.normal(
      scale=(1 / lrp.num_samples) * np.sqrt(0.5 / rho_grad), size=lrp.dim
  )
  dir_noisy = np.linalg.solve(hess_noisy, grad_noisy)
  dir_size = np.linalg.norm(np.linalg.solve(hess, grad))
  return w_cur - min(np.log(1 + dir_size) * (1 / dir_size), 1) * dir_noisy


def eigenclip(sym_mat, min_eval=1e-5):
  """operation of the eigenclip

  A = symmetric matrix
  min_eval = minimum eigenvalue for clipping

  return the modified matrix
  """
  eig_val, eig_vec = np.linalg.eigh(sym_mat)
  eval_mod = np.maximum(eig_val, min_eval * np.ones(eig_val.shape))
  clipped_mat = np.dot(eig_vec * eval_mod, eig_vec.T)
  return clipped_mat


def gd_priv(w_cur, lrp, hyper_dict):
  """Implementation of DP-GD.

  w = current point
  lr = logistic regression
  i = iteration number
  pb = auxillary information

  output is the next iterate
  """
  iters = hyper_dict["num_iteration"]
  inv_lr_gd = 0.25  # learning rate based on the smoothness
  sens = 1 / (lrp.num_samples * (inv_lr_gd))  # sensitivity
  rho = hyper_dict["total"] / iters  # divide total privacy budget up
  noise = np.random.normal(scale=sens / np.sqrt(2 * rho), size=lrp.dim)
  return w_cur - lrp.grad_wor(w_cur) / (inv_lr_gd) + noise


def sgd_priv(w_cur, lrp, hyper_dict):
  """Implementation of DP-SGD.

  w = current point
  lr = logistic regression
  i = iteration number
  pb = auxillary information

  output is the next iterate
  """
  batch_size = hyper_dict["batch_size"]
  sigma_privacy = hyper_dict["noise_multiplier"]
  lr_sgd = 4  # learning rate based on the smoothness
  sample_rate = batch_size / lrp.num_samples  # sampling probability
  sample_vec = np.random.binomial(n=1, p=sample_rate, size=lrp.num_samples)
  batch_idx = np.where(sample_vec == 1)[0]  # index of batch
  batch_size_t = len(batch_idx)
  noise = np.random.normal(scale=sigma_privacy, size=lrp.dim)
  grad_minibatch = lrp.grad_wor(
      w_cur, batch_idx
  )  # average gradient over batch_idx
  return w_cur - lr_sgd * (
      batch_size_t / batch_size * grad_minibatch + noise / batch_size
  )


def gd_priv_optls(w_cur, lrp, hyper_dict):
  """Implementation of DP-GD with back-tracking line search !!!

  this method is not private. We only use it as a baseline.

  w = current point
  lr = logistic regression
  i = iteration number
  pb = auxillary information

  output is the next iterate
  """
  iters = hyper_dict["num_iteration"]
  rho_grad = hyper_dict["total"] / iters  # divide total privacy budget up
  grad_scale = (1 / lrp.num_samples) * np.sqrt(0.5 / rho_grad)
  grad_noise = np.random.normal(scale=grad_scale, size=lrp.dim)
  dir_srch = lrp.grad(w_cur) + grad_noise
  stepsize_opt = backtracking_ls(lrp, dir_srch, w_cur)
  return w_cur - stepsize_opt * dir_srch


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
    w_cur = w_cur - step_size * dir
  if lrp.loss_wor(w_cur) < lrp.loss_wor(w_init):
    w_out = w_cur
  else:
    w_out = w_init
  return w_out


class DoubleNoiseMech:
  """Our Method: Double Noise Mechanism"""

  def __init__(self, lrp, type_reg="add", curvature_info="hessian"):
    """Initializer of the double noise mechanism

    lr = an instance of MyLogisticRegression
    type_reg = minimum eigenvalue modification type, it can be either 'add' or
    'clip'
    curvature_info = type of the second-order information
    """
    self.type_reg = type_reg
    self.curvature_info = curvature_info
    if self.curvature_info == "hessian":
      self.hess = lrp.hess_wor
    elif self.curvature_info == "ub":
      self.hess = lrp.upperbound_wor

  def update_rule(self, w_cur, lrp, hyper_dict):
    """Implementation of the double noise mechanism update rule--full batch"""
    noisy_grad_cur = self.noisy_grad(w_cur, lrp, hyper_dict)
    w_next = self.noisy_direction(w_cur, lrp, hyper_dict, noisy_grad_cur)
    return w_next

  def update_rule_stochastic(self, w_cur, lrp, hyper_dict):
    """Implementation of the double noise mechanism update rule--full batch"""
    noisy_grad_cur = self.noisy_grad(w_cur, lrp, hyper_dict, True)
    w_next = self.noisy_direction_stochastic(
        w_cur, lrp, hyper_dict, noisy_grad_cur
    )
    return w_next

  def noisy_grad(self, w_cur, lrp, hyper_dict, batch=False):
    """computing gradient"""
    if batch is False:
      rho_grad = (hyper_dict["grad_frac"] * hyper_dict["total"]) / hyper_dict[
          "num_iteration"
      ]
      noise_grad = np.random.normal(
          scale=(1 / lrp.num_samples) * np.sqrt(0.5 / rho_grad), size=lrp.dim
      )
      noisy_grad = lrp.grad(w_cur) + noise_grad
    else:
      std_grad = hyper_dict["noise_multiplier_grad"]
      pgrad = hyper_dict["batchsize_grad"] / lrp.num_samples
      sample_vec = np.random.binomial(n=1, p=pgrad, size=lrp.num_samples)
      batch_idx_grad = np.where(sample_vec == 1)[0]
      grad_minibatch = lrp.grad_wor(w_cur, batch_idx_grad)
      noise_grad = np.random.normal(scale=std_grad, size=lrp.dim)
      noisy_grad = (
          len(batch_idx_grad) / (lrp.num_samples * pgrad)
      ) * grad_minibatch + (noise_grad) / (lrp.num_samples * pgrad)
    return noisy_grad

  def noisy_direction(self, w_cur, lrp, hyper_dict, noisy_grad):
    """computing direction"""
    total = hyper_dict["total"]
    grad_frac = hyper_dict["grad_frac"]
    frac_trace = hyper_dict["trace_frac"]
    trace_coeff = hyper_dict["trace_coeff"]
    iters = hyper_dict["num_iteration"]
    rho_hess = (1 - grad_frac) * total / iters
    smooth_param = 0.25
    hess_cur = self.hess(w_cur)
    noisy_trace = trace_coeff * max(
        np.trace(hess_cur)
        + np.random.normal(
            scale=(0.25 / lrp.num_samples)
            * np.sqrt(0.5 / (frac_trace * rho_hess))
        ),
        0,
    )
    min_eval = max(
        (noisy_trace / ((lrp.num_samples) ** 2 * (1 - frac_trace) * rho_hess))
        ** (1 / 3),
        1 / (lrp.num_samples),
    )
    grad_norm = np.linalg.norm(noisy_grad)
    if self.type_reg == "add":  # Sensitivity is different for add vs clip
      sens2 = (
          grad_norm
          * smooth_param
          / (lrp.num_samples * min_eval**2 + smooth_param * min_eval)
      )
      noise2 = np.random.normal(
          scale=sens2 * np.sqrt(0.5 / ((1 - frac_trace) * rho_hess)),
          size=lrp.dim,
      )
      return (
          w_cur
          - np.linalg.solve(hess_cur + min_eval * np.eye(lrp.dim), noisy_grad)
          + noise2
      )
    # type_reg=clip
    sens2 = (
        grad_norm
        * smooth_param
        / (lrp.num_samples * min_eval**2 - smooth_param * min_eval)
    )
    noise2 = np.random.normal(
        scale=sens2 * np.sqrt(0.5 / ((1 - frac_trace) * rho_hess)), size=lrp.dim
    )
    eval_hess, evec_hess = np.linalg.eigh(hess_cur)
    eval_trunc = eval_hess[eval_hess >= min_eval]
    num_eig = len(eval_trunc)
    if num_eig == 0:
      hess_modified_inv = 1 / min_eval * np.eye(lrp.dim)
    else:
      evec_trun = evec_hess[:, -num_eig:]
      hess_modified_inv = np.dot(
          evec_trun * (1 / eval_trunc - 1 / min_eval * np.ones(num_eig)),
          evec_trun.T,
      ) + 1 / min_eval * np.eye(lrp.dim)
    return w_cur - (hess_modified_inv @ noisy_grad) + noise2

  def noisy_direction_stochastic(self, w_cur, lrp, hyper_dict, noisy_grad):
    """noisy direction for stochastic variant"""
    std_hess = hyper_dict["noise_multiplier_hess"]
    phess = hyper_dict["batchsize_hess"] / lrp.num_samples
    min_eval = hyper_dict["min_eval"]
    sample_vec = np.random.binomial(n=1, p=phess, size=lrp.num_samples)
    batch_idx_hess = np.where(sample_vec == 1)[0]
    batch_size_hess_t = len(batch_idx_hess)
    hess_cur = (
        (batch_size_hess_t)
        / (lrp.num_samples * phess)
        * self.hess(w_cur, batch_idx_hess)
    )
    smooth_param = 0.25  # smoothness parameter
    grad_norm = np.linalg.norm(noisy_grad)
    if self.type_reg == "add":  # Sensitivity is different for add vs clip
      sens2 = (
          grad_norm
          * smooth_param
          / (
              (lrp.num_samples * phess) * min_eval**2
              + smooth_param * min_eval
          )
      )
      noise2 = np.random.normal(scale=sens2 * std_hess, size=lrp.dim)
      return (
          w_cur
          - np.linalg.solve(
              hess_cur + min_eval * np.eye(len(hess_cur)), noisy_grad
          )
          + noise2
      )
    # type_reg=clip
    min_eval_c = max(min_eval, 1 / ((lrp.num_samples * phess)))
    sens2 = (
        grad_norm
        * smooth_param
        / (
            (lrp.num_samples * phess) * min_eval_c**2
            - smooth_param * min_eval_c
        )
    )
    noise2 = np.random.normal(scale=sens2 * std_hess, size=lrp.dim)
    eval_hess, evec_hess = np.linalg.eigh(hess_cur)
    eval_trunc = eval_hess[eval_hess >= min_eval_c]
    num_eig = len(eval_trunc)
    if num_eig == 0:
      hess_modified_inv = 1 / min_eval_c * np.eye(lrp.dim)
    else:
      evec_trun = evec_hess[:, -num_eig:]
      hess_modified_inv = np.dot(
          evec_trun * (1 / eval_trunc - 1 / min_eval_c * np.ones(num_eig)),
          evec_trun.T,
      ) + 1 / min_eval_c * np.eye(lrp.dim)
    return w_cur - (hess_modified_inv @ noisy_grad) + noise2

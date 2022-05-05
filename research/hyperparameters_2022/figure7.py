# Copyright 2022, The TensorFlow Privacy Authors.
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
"""Code for reproducing Figure 7 of paper."""

import math
import matplotlib.pyplot as plt
import numpy as np
import rdp_accountant

# pylint: disable=bare-except
# pylint: disable=g-import-not-at-top
# pylint: disable=g-multiple-import
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name

####################################################
# This file loads default values to reproduce
# figure 7 from the paper. If you'd like to
# provide your own value, modify the variables
# in the if statement controlled by this variable.
####################################################

load_values_to_reproduce_paper_fig = True


def repeat_logarithmic_rdp(orders, rdp, gamma):
  n = len(orders)
  assert len(rdp) == n
  assert min(orders) >= 1
  rdp_out = [None] * n
  for i in range(n):
    if orders[i] == 1:
      continue  # unfortunately the formula doesn't work in this case
    for j in range(n):
      # Compute (orders[i],eps)-RDP bound on A_gamma given that Q satisfies
      # (orders[i],rdp[i])-RDP and (orders[j],rdp[j])-RDP
      eps = rdp[i] + (
          1 - 1 / orders[j]) * rdp[j] + math.log(1 / gamma - 1) / orders[j] + (
              math.log(1 / gamma - 1) - math.log(math.log(1 / gamma))) / (
                  orders[i] - 1)
      if rdp_out[i] is None or eps < rdp_out[i]:
        rdp_out[i] = eps
  return rdp_out


def repeat_geometric_rdp(orders, rdp, gamma):
  n = len(orders)
  assert len(rdp) == n
  assert min(orders) >= 1
  rdp_out = [None] * n
  for i in range(n):
    if orders[i] == 1:
      continue  # formula doesn't work in this case
    for j in range(n):
      eps = rdp[i] + 2 * (1 - 1 / orders[j]) * rdp[j] + (
          2 / orders[j] + 1 / (orders[i] - 1)) * math.log(1 / gamma)
      if rdp_out[i] is None or eps < rdp_out[i]:
        rdp_out[i] = eps
  return rdp_out


def repeat_negativebinomial_rdp(orders, rdp, gamma, eta):
  n = len(orders)
  assert len(rdp) == n
  assert min(orders) >= 1
  assert 0 < gamma < 1
  assert eta > 0
  rdp_out = [None] * n
  # foo = log(eta/(1-gamma^eta))
  foo = math.log(eta) - math.log1p(-math.pow(gamma, eta))
  for i in range(n):
    if orders[i] == 1:
      continue  # forumla doesn't work for lambda=1
    for j in range(n):
      eps = rdp[i] + (1 + eta) * (1 - 1 / orders[j]) * rdp[j] - (
          (1 + eta) / orders[j] + 1 /
          (orders[i] - 1)) * math.log(gamma) + foo / (orders[i] - 1) + (
              1 + eta) * math.log1p(-gamma) / orders[j]
      if rdp_out[i] is None or eps < rdp_out[i]:
        rdp_out[i] = eps
  return rdp_out


def repeat_poisson_rdp(orders, rdp, tau):
  n = len(orders)
  assert len(rdp) == n
  assert min(orders) >= 1
  rdp_out = [None] * n
  for i in range(n):
    if orders[i] == 1:
      continue  # forumula doesn't work with lambda=1
    _, delta, _ = rdp_accountant.get_privacy_spent(
        orders, rdp, target_eps=math.log1p(1 / (orders[i] - 1)))
    rdp_out[i] = rdp[i] + tau * delta + math.log(tau) / (orders[i] - 1)
  return rdp_out


if load_values_to_reproduce_paper_fig:
  from figure7_default_values import orders, rdp, lr_acc, num_trials, lr_rates, gammas, non_private_acc
else:
  orders = []  # Complete with the list of orders
  rdp = []  # Complete with the list of RDP
  lr_acc = {}  # Complete with a dictionary such that keys
  # are learning rates and values are the
  # corresponding model's accuracy
  num_trials = 1000  # num_trials to average results over
  lr_rates = np.asarray([])  # 1D array of learning rate candidates
  gammas = np.asarray(
      [])  # 1D array of gamma parameters to the random distributions.
  non_private_acc = 1.  # accuracy of a non-private run (for plotting only)

for dist_id in range(4):
  res_x = np.zeros_like(gammas)
  res_y = np.zeros_like(res_x)
  res_y_max = non_private_acc * np.ones_like(res_x)
  for gamma_id, gamma in enumerate(gammas):
    expected = (1 / gamma - 1) / np.log(1 / gamma)
    best_acc_trials = []
    for trial in range(num_trials):
      if dist_id == 0:
        K = np.random.logseries(1 - gamma)
        label = 'logarithmic distribution $\\eta=0$'
        color = 'b'
        eps = repeat_logarithmic_rdp(orders, rdp, gamma)
      elif dist_id == 1:
        if load_values_to_reproduce_paper_fig and gamma < 1e-4:
          continue
        K = np.random.geometric(gamma)
        label = 'geometric distribution $\\eta=1$'
        color = 'g'
        eps = repeat_geometric_rdp(orders, rdp, gamma)
      elif dist_id == 2:
        if load_values_to_reproduce_paper_fig and gamma < 1e-07:
          continue
        eta = 0.5
        K = 0
        while K == 0:
          K = np.random.negative_binomial(eta, gamma)
        label = 'negative binomial $\\eta=0.5$'
        color = 'k'
        eps = repeat_negativebinomial_rdp(orders, rdp, gamma, eta)
      elif dist_id == 3:
        if load_values_to_reproduce_paper_fig and gamma < 0.0015:
          continue
        gamma_factor = 100
        K = np.random.poisson(gamma * gamma_factor)
        label = 'poisson distribution'
        color = 'm'
        eps = repeat_poisson_rdp(orders, rdp, gamma * gamma_factor)
      best_acc = 0.
      best_lr = -1.
      for k in range(K):
        # pick a hyperparam candidate uniformly at random
        j = np.random.randint(0, len(lr_rates))
        lr_candidate = lr_rates[j]
        try:
          acc = lr_acc[str(lr_candidate)]
        except:
          print('lr - acc pair missing for ' + str(lr_candidate))
          acc = 0.
        if best_acc < acc:
          best_acc = acc
          best_lr = lr_candidate
      best_acc_trials.append(best_acc)
    try:
      res_x[gamma_id] = np.min(eps)
      res_y[gamma_id] = np.mean(best_acc_trials)
    except:
      print('skipping ' + str(gamma_id))
  if dist_id == 0:
    plt.hlines(
        res_y_max[0],
        xmin=-1.,
        xmax=20.,
        color='r',
        label='baseline (non-private search)')
  if dist_id >= 1:
    res_x = res_x[2:]
    res_y = res_y[2:]
  plt.plot(res_x, res_y, label=label, color=color)

if load_values_to_reproduce_paper_fig:
  plt.xlim([0.5, 8.])
  plt.ylim([0.85, 0.97])
plt.xlabel('Privacy budget')
plt.ylabel('Model Accuracy for Best Hyperparameter')
plt.legend(loc='lower right')
plt.savefig('rdp_hyper_search.pdf', bbox_inches='tight')

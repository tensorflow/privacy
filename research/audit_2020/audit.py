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
"""Class for running auditing procedure."""

import numpy as np
from statsmodels.stats import proportion

import attacks

def compute_results(poison_scores, unpois_scores, pois_ct,
                    alpha=0.05, threshold=None):
  """
  Searches over thresholds for the best epsilon lower bound and accuracy.
  poison_scores: list of scores from poisoned models
  unpois_scores: list of scores from unpoisoned models
  pois_ct: number of poison points
  alpha: confidence parameter
  threshold: if None, search over all thresholds, else use given threshold
  """
  if threshold is None:  # search for best threshold
    all_thresholds = np.unique(poison_scores + unpois_scores)
  else:
    all_thresholds = [threshold]

  poison_arr = np.array(poison_scores)
  unpois_arr = np.array(unpois_scores)

  best_threshold, best_epsilon, best_acc = None, 0, 0
  for thresh in all_thresholds:
    epsilon, acc = compute_epsilon_and_acc(poison_arr, unpois_arr, thresh,
                                           alpha, pois_ct)
    if epsilon > best_epsilon:
      best_epsilon, best_threshold = epsilon, thresh
    best_acc = max(best_acc, acc)
  return best_threshold, best_epsilon, best_acc


def compute_epsilon_and_acc(poison_arr, unpois_arr, threshold, alpha, pois_ct):
  """For a given threshold, compute epsilon and accuracy."""
  poison_ct = (poison_arr > threshold).sum()
  unpois_ct = (unpois_arr > threshold).sum()

  # clopper_pearson uses alpha/2 budget on upper and lower
  # so total budget will be 2*alpha/2 = alpha
  p1, _ = proportion.proportion_confint(poison_ct, poison_arr.size,
                                        alpha, method='beta')
  _, p0 = proportion.proportion_confint(unpois_ct, unpois_arr.size,
                                        alpha, method='beta')

  if (p1 <= 1e-5) or (p0 >= 1 - 1e-5):  # divide by zero issues
    return 0, 0

  if (p0 + p1) > 1:  # see Appendix A
    p0, p1 = (1-p1), (1-p0)

  epsilon = np.log(p1/p0)/pois_ct
  acc = (p1 + (1-p0))/2  # this is not necessarily the best accuracy

  return epsilon, acc


class AuditAttack(object):
  """Audit attack class. Generates poisoning, then runs auditing algorithm."""
  def __init__(self, train_x, train_y, train_function):
    """
    train_x: training features
    train_y: training labels
    name: identifier for the attack
    train_function: function returning membership score
    """
    self.train_x, self.train_y = train_x, train_y
    self.train_function = train_function
    self.poisoning = None

  def make_poisoning(self, pois_ct, attack_type, l2_norm=10):
    """Get poisoning data."""
    return attacks.make_many_poisoned_datasets(self.train_x, self.train_y, [pois_ct],
                                  attack=attack_type, l2_norm=l2_norm)

  def run_experiments(self, num_trials):
    """Runs all training experiments."""
    (pois_x1, pois_y1), (pois_x2, pois_y2) = self.poisoning['data']
    sample_x, sample_y = self.poisoning['pois']

    poison_scores = []
    unpois_scores = []

    for i in range(num_trials):
      poison_tuple = (pois_x1, pois_y1, sample_x, sample_y, i)
      unpois_tuple = (pois_x2, pois_y2, sample_x, sample_y, num_trials + i)
      poison_scores.append(self.train_function(poison_tuple))
      unpois_scores.append(self.train_function(unpois_tuple))

    return poison_scores, unpois_scores

  def run(self, pois_ct, attack_type, num_trials, alpha=0.05,
          threshold=None, l2_norm=10):
    """Complete auditing algorithm. Generates poisoning if necessary."""
    if self.poisoning is None:
      self.poisoning = self.make_poisoning(pois_ct, attack_type, l2_norm)
      self.poisoning['data'] = self.poisoning[pois_ct]

    poison_scores, unpois_scores = self.run_experiments(num_trials)

    results = compute_results(poison_scores, unpois_scores, pois_ct,
                              alpha=alpha, threshold=threshold)
    return results

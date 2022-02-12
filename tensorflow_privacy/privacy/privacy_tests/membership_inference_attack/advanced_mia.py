# Copyright 2022, The TensorFlow Authors.
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
"""Functions for advanced membership inference attacks."""

import functools
from typing import Sequence, Union
import numpy as np
import scipy.stats
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.utils import log_loss


def replace_nan_with_column_mean(a: np.ndarray):
  """Replaces each NaN with the mean of the corresponding column."""
  mean = np.nanmean(a, axis=0)  # get the column-wise mean
  for i in range(a.shape[1]):
    np.nan_to_num(a[:, i], copy=False, nan=mean[i])


def compute_score_offset(stat_target: Union[np.ndarray, Sequence[float]],
                         stat_in: Sequence[np.ndarray],
                         stat_out: Sequence[np.ndarray],
                         option: str = 'both',
                         median_or_mean: str = 'median') -> np.ndarray:
  """Computes score of each sample as stat_target - some offset.

  Args:
    stat_target: a list or numpy array where stat_target[i] is the statistics of
      example i computed from the target model. stat_target[i] is an array of k
      scalars for k being the number of augmentations for each sample.
    stat_in: a list where stat_in[i] is the in-training statistics of example i.
      stat_in[i] is a m by k numpy array where m is the number of shadow models
      and k is the number of augmentations for each sample. m can be different
      for different examples.
    stat_out: a list where stat_out[i] is the out-training statistics of example
      i. stat_out[i] is a m by k numpy array where m is the number of shadow
      models and k is the number of augmentations for each sample.
    option: using stat_in ("in"), stat_out ("out"), or both ("both").
    median_or_mean: use median or mean across shadow models.

  Returns:
    The score of each sample as stat_target - some offset, where offset is
    computed with stat_in, stat_out, or both depending on the option.
    The relation between the score and the membership depends on that
    between stat_target and membership.
  """
  if option not in ['both', 'in', 'out']:
    raise ValueError('option should be "both", "in", or "out".')
  if median_or_mean not in ['median', 'mean']:
    raise ValueError('median_or_mean should be either "median" or "mean".')
  if option in ['in', 'both']:
    if any([s.ndim != 2 for s in stat_in]):
      raise ValueError('Each element in stat_in should be a 2-d numpy array.')
    if any([s.shape[1] != stat_in[0].shape[1] for s in stat_in]):
      raise ValueError('Each element in stat_in should have the same size '
                       'in the second dimension.')
  if option in ['out', 'both']:
    if any([s.ndim != 2 for s in stat_out]):
      raise ValueError('Each element in stat_out should be a 2-d numpy array.')
    if any([s.shape[1] != stat_out[0].shape[1] for s in stat_out]):
      raise ValueError('Each element in stat_out should have the same size '
                       'in the second dimension.')
  func_avg = functools.partial(
      np.nanmedian if median_or_mean == 'median' else np.nanmean, axis=0)

  if option == 'both':  # use the average of the in-score and out-score
    avg_in = np.array(list(map(func_avg, stat_in)))
    avg_out = np.array(list(map(func_avg, stat_out)))
    # use average in case of NaN
    replace_nan_with_column_mean(avg_in)
    replace_nan_with_column_mean(avg_out)
    offset = (avg_in + avg_out) / 2
  elif option == 'in':  # use in-score only
    offset = np.array(list(map(func_avg, stat_in)))
    replace_nan_with_column_mean(offset)
  else:  # use out-score only
    offset = np.array(list(map(func_avg, stat_out)))
    replace_nan_with_column_mean(offset)
  scores = (stat_target - offset).mean(axis=1)
  return scores


def compute_score_lira(stat_target: Union[np.ndarray, Sequence[float]],
                       stat_in: Sequence[np.ndarray],
                       stat_out: Sequence[np.ndarray],
                       option: str = 'both',
                       fix_variance: bool = False,
                       median_or_mean: str = 'median') -> np.ndarray:
  """Computes score of each sample using Gaussian distribution fitting.

  Args:
    stat_target: a list or numpy array where stat_target[i] is the statistics of
      example i computed from the target model. stat_target[i] is an array of k
      scalars for k being the number of augmentations for each sample.
    stat_in: a list where stat_in[i] is the in-training statistics of example i.
      stat_in[i] is a m by k numpy array where m is the number of shadow models
      and k is the number of augmentations for each sample. m can be different
      for different examples.
    stat_out: a list where stat_out[i] is the out-training statistics of example
      i. stat_out[i] is a m by k numpy array where m is the number of shadow
      models and k is the number of augmentations for each sample.
    option: using stat_in ("in"), stat_out ("out"), or both ("both").
    fix_variance: whether to use the same variance for all examples.
    median_or_mean: use median or mean across shadow models.

  Returns:
    log(Pr(out)) - log(Pr(in)), log(Pr(out)), or -log(Pr(in)) depending on the
    option. In-training sample is expected to have small value.
    The idea is from https://arxiv.org/pdf/2112.03570.pdf.
  """
  # median of statistics across shadow models
  if option not in ['both', 'in', 'out']:
    raise ValueError('option should be "both", "in", or "out".')
  if median_or_mean not in ['median', 'mean']:
    raise ValueError('median_or_mean should be either "median" or "mean".')
  if option in ['in', 'both']:
    if any([s.ndim != 2 for s in stat_in]):
      raise ValueError('Each element in stat_in should be a 2-d numpy array.')
    if any([s.shape[1] != stat_in[0].shape[1] for s in stat_in]):
      raise ValueError('Each element in stat_in should have the same size '
                       'in the second dimension.')
  if option in ['out', 'both']:
    if any([s.ndim != 2 for s in stat_out]):
      raise ValueError('Each element in stat_out should be a 2-d numpy array.')
    if any([s.shape[1] != stat_out[0].shape[1] for s in stat_out]):
      raise ValueError('Each element in stat_out should have the same size '
                       'in the second dimension.')

  func_avg = functools.partial(
      np.nanmedian if median_or_mean == 'median' else np.nanmean, axis=0)
  if option in ['in', 'both']:
    avg_in = np.array(list(map(func_avg, stat_in)))  # n by k array
    replace_nan_with_column_mean(avg_in)  # use column average in case of NaN
  if option in ['out', 'both']:
    avg_out = np.array(list(map(func_avg, stat_out)))
    replace_nan_with_column_mean(avg_out)

  if fix_variance:
    # standard deviation of statistics across shadow models and examples
    if option in ['in', 'both']:
      std_in = np.nanstd(
          np.concatenate([l - m[np.newaxis] for l, m in zip(stat_in, avg_in)]))
    if option in ['out', 'both']:
      std_out = np.nanstd(
          np.concatenate([l - m[np.newaxis] for l, m in zip(stat_out, avg_out)
                         ]))
  else:
    # standard deviation of statistics across shadow models
    func_std = functools.partial(np.nanstd, axis=0)
    if option in ['in', 'both']:
      std_in = np.array(list(map(func_std, stat_in)))
      replace_nan_with_column_mean(std_in)
    if option in ['out', 'both']:
      std_out = np.array(list(map(func_std, stat_out)))
      replace_nan_with_column_mean(std_out)

  stat_target = np.array(stat_target)
  if option in ['in', 'both']:
    log_pr_in = scipy.stats.norm.logpdf(stat_target, avg_in, std_in + 1e-30)
  if option in ['out', 'both']:
    log_pr_out = scipy.stats.norm.logpdf(stat_target, avg_out, std_out + 1e-30)

  if option == 'both':
    scores = -(log_pr_in - log_pr_out).mean(axis=1)
  elif option == 'in':
    scores = -log_pr_in.mean(axis=1)
  else:
    scores = log_pr_out.mean(axis=1)
  return scores


def convert_logit_to_prob(logit: np.ndarray) -> np.ndarray:
  """Converts logits to probability vectors.

  Args:
    logit: n by c array where n is the number of samples and c is the number of
      classes.

  Returns:
    The probability vectors as n by c array
  """
  prob = logit - np.max(logit, axis=1, keepdims=True)
  prob = np.array(np.exp(prob), dtype=np.float64)
  prob = prob / np.sum(prob, axis=1, keepdims=True)
  return prob


def calculate_statistic(pred: np.ndarray,
                        labels: np.ndarray,
                        is_logits: bool = True,
                        option: str = 'logit',
                        small_value: float = 1e-45):
  """Calculates the statistics of each sample.

  The statistics is:
    for option="conf with prob", p, the probability of the true class;
    for option="xe", the cross-entropy loss;
    for option="logit", log(p / (1 - p));
    for option="conf with logit", max(logits);
    for option="hinge", logit of the true class - max(logits of the other
    classes).

  Args:
    pred: the logits or probability vectors, depending on the value of is_logit.
      An array of size n by c where n is the number of samples and c is the
      number of classes
    labels: true labels of samples (integer valued)
    is_logits: whether pred is logits or probability vectors
    option: confidence using probability, xe loss, logit of confidence,
      confidence using logits, hinge loss
    small_value: a small value to avoid numerical issue

  Returns:
    the computed statistics as size n array
  """
  if option not in [
      'conf with prob', 'xe', 'logit', 'conf with logit', 'hinge'
  ]:
    raise ValueError(
        'option should be one of ["conf with prob", "xe", "logit", "conf with logit", "hinge"].'
    )
  if option in ['conf with logit', 'hinge']:
    if not is_logits:  # the input needs to be the logits
      raise ValueError('To compute statistics with option "conf with logit" '
                       'or "hinge", the input must be logits instead of '
                       'probability vectors.')
  elif is_logits:
    pred = convert_logit_to_prob(pred)

  n = labels.size  # number of samples
  if option in ['conf with prob', 'conf with logit']:
    return pred[range(n), labels]
  if option == 'xe':
    return log_loss(labels, pred)
  if option == 'logit':
    p_true = pred[range(n), labels]
    pred[range(n), labels] = 0
    p_other = pred.sum(axis=1)
    return np.log(p_true + small_value) - np.log(p_other + small_value)
  if option == 'hinge':
    l_true = pred[range(n), labels]
    pred[range(n), labels] = -np.inf
    return l_true - pred.max(axis=1)
  raise ValueError

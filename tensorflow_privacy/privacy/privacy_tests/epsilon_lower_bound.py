# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various functions to convert MIA or secret sharer to epsilon lower bounds."""

import enum
import numbers
from typing import Dict, Iterable, Optional, Sequence, Union

import immutabledict
import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.optimize
import scipy.stats
import sklearn.metrics
from statsmodels.stats import proportion


def _get_tp_fp_for_thresholds(pos_scores: np.ndarray,
                              neg_scores: np.ndarray,
                              thresholds: Optional[np.ndarray] = None):
  """Gets all the tp and fp for a given array of thresholds.

  Args:
    pos_scores: per-example scores for the positive class.
    neg_scores: per-example scores for the negative class.
    thresholds: an array of thresholds to consider. Will consider elements
      **above** as positive. If not provided, will enumerate through all
      possible thresholds.

  Returns:
    A tuple as the true positives and false positives.
  """
  if thresholds is None:
    # pylint:disable=protected-access
    fp, tp, _ = sklearn.metrics._ranking._binary_clf_curve(
        y_true=np.concatenate([
            np.ones_like(pos_scores, dtype=int),
            np.zeros_like(neg_scores, dtype=int)
        ]),
        y_score=np.concatenate([pos_scores, neg_scores]))
    return tp, fp

  def get_cum_sum(scores, thresholds):
    values = np.concatenate([scores, thresholds])
    indicators = np.concatenate(
        [np.ones_like(scores, dtype=int),
         np.zeros_like(thresholds, dtype=int)])
    sort_idx = np.argsort(values)[::-1]  # Descending
    indicators = indicators[sort_idx]
    return np.cumsum(indicators)[indicators == 0]

  tp = get_cum_sum(pos_scores, thresholds)
  fp = get_cum_sum(neg_scores, thresholds)
  return tp, fp


class BoundMethod(enum.Enum):
  """Methods to use for bound of ratio of binomial proportions."""
  KATZ_LOG = 'katz-log'
  ADJUSTED_LOG = 'adjusted-log'
  BAILEY = 'bailey'
  INV_SINH = 'inv-sinh'
  CLOPPER_PEARSON = 'clopper-pearson'


class EpsilonLowerBound:
  """Differential privacy (DP) epsilon lower bound.

  This class computes a statistical epsilon lower bound by looking at the log
  ratio of tpr and fpr. The tpr / fpr ratio bound is from `RatioBound` class.

  For example, in membership inference attack, the attacker sets a threshold and
  predicts samples with top probability larger than the thresholds as member.
  If the model is trained withs DP guarantee, then we should expect
  log(tpr / fpr) <= epsilon, where tpr and fpr are the true positive and false
  positive rates of the attacker. Therefore, we can use log(tpr / fpr) to derive
  an epsilon lower bound.

  The idea of using Clopper Pearson for estimating epsilon lower bound is from
  https://arxiv.org/pdf/2006.07709.pdf.
  The idea of using log Katz is from https://arxiv.org/pdf/2210.08643.pdf.

  Examples:
    >>> lb = elb.EpsilonLowerBound(train_top_probs, test_top_probs, alpha=0.05)
    >>> methods = [BoundMethod.BAILEY, BoundMethod.KATZ_LOG]
    >>> lb.compute_epsilon_lower_bounds(methods, k=5)
  """

  def __init__(self,
               pos_scores: np.ndarray,
               neg_scores: np.ndarray,
               alpha: float,
               two_sided_threshold: bool = True,
               thresholds: Optional[np.ndarray] = None):
    """Initializes the epsilon lower bound class.

    Args:
      pos_scores: per-example scores for the positive class.
      neg_scores: per-example scores for the negative class.
      alpha: the confidence level, must be < 0.5.
      two_sided_threshold: if False, will consider thresholds such that elements
        **above** are predicted as positive, i.e., tpr / fpr and tnr / fnr. If
        True, will also consider fpr / tpr and fnr / tnr.
      thresholds: an array of thresholds to consider. If not provided, will
        enumerate through all possible thresholds.
    """
    if pos_scores.ndim != 1:
      raise ValueError('pos_score should be a 1-dimensional array, '
                       f'but got {pos_scores.ndim}.')
    if neg_scores.ndim != 1:
      raise ValueError('pos_score should be a 1-dimensional array, '
                       f'but got {neg_scores.ndim}.')
    if alpha >= 0.5:
      raise ValueError('alpha should be < 0.5, e.g. alpha=0.05, '
                       f'but got {alpha}.')

    pos_size, neg_size = pos_scores.size, neg_scores.size
    tp, fp = _get_tp_fp_for_thresholds(pos_scores, neg_scores, thresholds)
    fn, tn = pos_size - tp, neg_size - fp

    # We consider both tpr / fpr and tnr / fnr.
    self._rbs = [
        RatioBound(tp, fp, pos_size, neg_size, alpha),
        RatioBound(tn, fn, neg_size, pos_size, alpha)
    ]
    if two_sided_threshold:
      self._rbs.extend([
          # pylint: disable-next=arguments-out-of-order
          RatioBound(fp, tp, neg_size, pos_size, alpha),
          RatioBound(fn, tn, pos_size, neg_size, alpha)
      ])

  def compute_epsilon_lower_bound(self,
                                  method: BoundMethod,
                                  k: Optional[int] = None
                                 ) -> npt.NDArray[float]:
    """Computes lower bound w/ a specified method and returns top-k epsilons.

    Args:
      method: the method to use for ratio bound.
      k: if specified, will return top-k values.

    Returns:
      An array of bounds.
    """
    if method not in self._rbs[0].available_methods:
      raise ValueError(f'Method {method} not recognized.')
    ratio_bound = np.concatenate([rb.compute_bound(method) for rb in self._rbs])
    bounds = np.log(ratio_bound[ratio_bound > 0])
    bounds = np.sort(bounds)[::-1]
    if k is None or k >= bounds.size:
      return bounds
    return bounds[:k]

  def compute_epsilon_lower_bounds(
      self,
      methods: Optional[Iterable[BoundMethod]] = None,
      k: Optional[int] = None) -> Dict[BoundMethod, npt.NDArray[float]]:
    """Computes lower bounds with all methods and returns the top-k epsilons.

    Args:
      methods: the methods to use for ratio bound. If not specified, will use
        all available methods.
      k: if specified, will return top-k values for each method.

    Returns:
      A dictionary, mapping method to the corresponding bound array.
    """
    return {
        method: self.compute_epsilon_lower_bound(method, k)
        for method in methods or self._rbs[0].available_methods.keys()
    }


class RatioBound:
  """Lower bound of ratio of binomial proportions.

  This class implements several methods to compute a statistical lower bound of
  the ratio of binomial proportions, e.g. tpr / fpr.
  Most of the methods are based on https://doi.org/10.1111/2041-210X.12304 and
  their code at https://CRAN.R-project.org/package=asbio.
  Clopper pearson is based on https://arxiv.org/pdf/2006.07709.pdf.

  Examples:
    >>> tp, fp = np.array([100, 90]), np.array([10, 5])
    >>> pos_size, neg_size = 110, 80
    >>> rb = elb.RatioBound(tp, fp, pos_size, neg_size, 0.05)
    >>> rb.compute_bound(BoundMethod.BAILEY)
    array([4.61953896, 6.87647915])
    >>> rb.compute_bounds([BoundMethod.BAILEY, BoundMethod.KATZ_LOG])
    {<BoundMethod.BAILEY: 'bailey'>: array([4.61953896, 6.87647915]),
     <BoundMethod.KATZ_LOG: 'katz-log'>: array([4.45958661, 6.39712581])}

  Attributes:
    available_methods: a dictionary mapping BoundMethod to the function.
  """

  def __init__(self, tp: Union[Sequence[int], int], fp: Union[Sequence[int],
                                                              int],
               pos_size: int, neg_size: int, alpha: float):
    """Initializes the ratio bound class.

    Args:
      tp: true positives.
      fp: false positives. Should be of the same length as tp.
      pos_size: number of real positive samples.
      neg_size: number of real negative samples.
      alpha: the confidence level, must be < 0.5.
    """
    if alpha >= 0.5:
      raise ValueError('alpha should be < 0.5, e.g. alpha=0.05, '
                       f'but got {alpha}.')
    self._is_scalar = False  # Would return scalar if `tp` is a scalar.
    # Convert tp or fp to list if it is a scalar.
    if isinstance(tp, numbers.Number):
      tp = [tp]
      self._is_scalar = True
    if isinstance(fp, numbers.Number):
      fp = [fp]
    if len(tp) != len(fp):
      raise ValueError('tp and fp should have the same number of elements, '
                       f'but get {len(tp)} and {len(fp)} respectively.')
    # Some methods need the original values.
    self._tp_orig = np.array(tp, dtype=float)
    self._fp_orig = np.array(fp, dtype=float)
    if np.any(self._tp_orig > pos_size) or np.any(self._tp_orig < 0):
      raise ValueError('tp needs to be in [0, pos_size].')
    if np.any(self._fp_orig > neg_size) or np.any(self._fp_orig < 0):
      raise ValueError('fp needs to be in [0, neg_size].')

    self.available_methods = immutabledict.immutabledict({
        BoundMethod.KATZ_LOG: self._bound_katz_log,
        BoundMethod.ADJUSTED_LOG: self._bound_adjusted_log,
        BoundMethod.BAILEY: self._bound_bailey,
        BoundMethod.INV_SINH: self._bound_inv_hyperbolic_sine,
        BoundMethod.CLOPPER_PEARSON: self._bound_clopper_pearson,
    })
    self._alpha = alpha
    self._z = scipy.stats.norm.ppf(alpha)
    self._pos_size, self._neg_size = pos_size, neg_size

    # Some methods need to adjust maximum possible values. We record the
    # adjusted arrays.
    idx_max = np.logical_and(self._tp_orig == self._pos_size,
                             self._fp_orig == self._neg_size)
    self._tp = np.where(idx_max, self._pos_size - 0.5, self._tp_orig)
    self._fp = np.where(idx_max, self._neg_size - 0.5, self._fp_orig)

    # Some methods need to handle 0 specifically. We record the indices.
    self._idx_tp_0, self._idx_fp_0 = (self._tp == 0), (self._fp == 0)

  def _get_statistics(self, tp, fp):
    """Returns tpr, fpr, fnr, tnr for given tp, fp."""
    tpr, fpr = tp / self._pos_size, fp / self._neg_size
    fnr, tnr = 1 - tpr, 1 - fpr
    return tpr, fpr, fnr, tnr

  def compute_bound(self,
                    method: BoundMethod) -> Union[float, npt.NDArray[float]]:
    """Computes ratio bound using a specified method.

    Args:
      method: the method to use for ratio bound.

    Returns:
      An array of bounds or a scalar if the input tp is scalar.
    """
    if method not in self.available_methods:
      raise ValueError(f'Method {method} not recognized.')
    bound = self.available_methods[method]()
    if self._is_scalar:
      bound = bound[0]  # Take the element if of size 1
    return bound

  def compute_bounds(
      self,
      methods: Optional[Iterable[BoundMethod]] = None
  ) -> Dict[BoundMethod, Union[float, npt.NDArray[float]]]:
    """Computes ratio bounds for specified methods.

    Args:
      methods: the methods to use for ratio bound. If not specified, will use
        all available methods.

    Returns:
      A dictionary, mapping method to the corresponding bound.
    """
    return {
        method: self.compute_bound(method)
        for method in methods or self.available_methods.keys()
    }

  def _bound_katz_log(self) -> npt.NDArray[float]:
    """Uses the logarithm Katz method to compute lower bound of ratio."""
    tp, fp = self._tp, np.where(self._idx_fp_0, 0.5, self._fp)
    tpr, fpr, fnr, tnr = self._get_statistics(tp, fp)
    empirical_ratio = tpr / fpr
    sqrt_term = np.sqrt(fnr / tp + tnr / fp)
    return np.where(self._idx_tp_0, 0,
                    empirical_ratio * np.exp(self._z * sqrt_term))

  def _bound_adjusted_log(self) -> npt.NDArray[float]:
    """Uses the logarithm Walters method to compute lower bound of ratio."""
    log_empirical_ratio = (
        np.log((self._tp + 0.5) / (self._pos_size + 0.5)) - np.log(
            (self._fp + 0.5) / (self._neg_size + 0.5)))
    sqrt_term = np.sqrt(1 / (self._tp + 0.5) - 1 / (self._pos_size + 0.5) + 1 /
                        (self._fp + 0.5) - 1 / (self._neg_size + 0.5))
    return np.where(
        np.logical_and(self._idx_tp_0, self._idx_fp_0), 0,
        np.exp(log_empirical_ratio) * np.exp(self._z * sqrt_term))

  def _bound_bailey(self) -> npt.NDArray[float]:
    """Uses the Bailey method to compute lower bound of ratio."""
    tp = np.where(self._tp_orig == self._pos_size, self._pos_size - 0.5,
                  self._tp_orig)
    fp = np.where(self._fp_orig == self._neg_size, self._neg_size - 0.5,
                  self._fp_orig)
    fp[self._idx_fp_0] = 0.5
    tpr, fpr, fnr, tnr = self._get_statistics(tp, fp)
    empirical_ratio = tpr / fpr
    power_3_term_numer = 1 + self._z / 3 * np.sqrt(fnr / tp + tnr / fp -
                                                   (self._z**2 * fnr * tnr) /
                                                   (9 * tp * fp))
    power_3_term_denom = 1 - (self._z**2 * tnr) / (9 * fp)
    return np.where(
        self._idx_tp_0, 0,
        empirical_ratio * (power_3_term_numer / power_3_term_denom)**3)

  def _bound_inv_hyperbolic_sine(self) -> npt.NDArray[float]:
    """Uses the inverse sinh method to compute lower bound of ratio."""
    tp, fp = self._tp, np.where(self._idx_fp_0, self._z**2, self._fp)
    empirical_ratio = (tp / fp) / (self._pos_size / self._neg_size)
    in_inve_sinh = self._z / 2 * np.sqrt(1 / tp - 1 / self._pos_size + 1 / fp -
                                         1 / self._neg_size)
    return np.where(self._idx_tp_0, 0,
                    empirical_ratio * np.exp(2 * np.arcsinh(in_inve_sinh)))

  def _bound_clopper_pearson(self) -> npt.NDArray[float]:
    """Uses the Clopper-Pearson method to compute lower bound of ratio."""
    # proportion_confint uses alpha / 2 budget on upper and lower, so total
    # budget will be 2 * alpha/2 = alpha.
    p1, _ = proportion.proportion_confint(
        self._tp_orig, self._pos_size, self._alpha, method='beta')
    _, p0 = proportion.proportion_confint(
        self._fp_orig, self._neg_size, self._alpha, method='beta')
    # Handles divide by zero issues
    return np.where(np.logical_or(p1 <= 0, p0 >= 1), 0, p1 / p0)

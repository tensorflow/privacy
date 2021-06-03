# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""RDP analysis of the Sampled Gaussian Mechanism.

Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:

  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
from scipy import special
import six


########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative.")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx


def _log_sub_sign(logx, logy):
  """Returns log(exp(logx)-exp(logy)) and its sign."""
  if logx > logy:
    s = True
    mag = logx + np.log(1 - np.exp(logy - logx))
  elif logx < logy:
    s = False
    mag = logy + np.log(1 - np.exp(logx - logy))
  else:
    s = True
    mag = -np.inf

  return s, mag


def _log_print(logx):
  """Pretty print."""
  if logx < math.log(sys.float_info.max):
    return "{}".format(math.exp(logx))
  else:
    return "exp({})".format(logx)


def _log_comb(n, k):
  return (special.gammaln(n + 1) -
          special.gammaln(k + 1) - special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, six.integer_types)

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  """Compute log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)


def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.

  Returns:
    Pair of (delta, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if eps < 0:
    raise ValueError("Value of privacy loss bound epsilon must be >=0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   delta = min( np.exp((rdp_vec - eps) * (orders_vec - 1)) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
  logdeltas = []  # work in log space to avoid overflows
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1: raise ValueError("Renyi divergence order must be >=1.")
    if r < 0: raise ValueError("Renyi divergence must be >=0.")
    # For small alpha, we are better of with bound via KL divergence:
    # delta <= sqrt(1-exp(-KL)).
    # Take a min of the two bounds.
    logdelta = 0.5*math.log1p(-math.exp(-r))
    if a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value for alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      rdp_bound = (a - 1) * (r - eps + math.log1p(-1/a)) - math.log(a)
      logdelta = min(logdelta, rdp_bound)

    logdeltas.append(logdelta)

  idx_opt = np.argmin(logdeltas)
  return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1: raise ValueError("Renyi divergence order must be >=1.")
    if r < 0: raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def _stable_inplace_diff_in_log(vec, signs, n=-1):
  """Replaces the first n-1 dims of vec with the log of abs difference operator.

  Args:
    vec: numpy array of floats with size larger than 'n'
    signs: Optional numpy array of bools with the same size as vec in case one
      needs to compute partial differences vec and signs jointly describe a
      vector of real numbers' sign and abs in log scale.
    n: Optonal upper bound on number of differences to compute. If negative, all
      differences are computed.

  Returns:
    The first n-1 dimension of vec and signs will store the log-abs and sign of
    the difference.

  Raises:
    ValueError: If input is malformed.
  """

  assert vec.shape == signs.shape
  if n < 0:
    n = np.max(vec.shape) - 1
  else:
    assert np.max(vec.shape) >= n + 1
  for j in range(0, n, 1):
    if signs[j] == signs[j + 1]:  # When the signs are the same
      # if the signs are both positive, then we can just use the standard one
      signs[j], vec[j] = _log_sub_sign(vec[j + 1], vec[j])
      # otherwise, we do that but toggle the sign
      if not signs[j + 1]:
        signs[j] = ~signs[j]
    else:  # When the signs are different.
      vec[j] = _log_add(vec[j], vec[j + 1])
      signs[j] = signs[j + 1]


def _get_forward_diffs(fun, n):
  """Computes up to nth order forward difference evaluated at 0.

  See Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf

  Args:
    fun: Function to compute forward differences of.
    n: Number of differences to compute.

  Returns:
    Pair (deltas, signs_deltas) of the log deltas and their signs.
  """
  func_vec = np.zeros(n + 3)
  signs_func_vec = np.ones(n + 3, dtype=bool)

  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
  deltas = np.zeros(n + 2)
  signs_deltas = np.zeros(n + 2, dtype=bool)
  for i in range(1, n + 3, 1):
    func_vec[i] = fun(1.0 * (i - 1))
  for i in range(0, n + 2, 1):
    # Diff in log scale
    _stable_inplace_diff_in_log(func_vec, signs_func_vec, n=n + 2 - i)
    deltas[i] = func_vec[0]
    signs_deltas[i] = signs_func_vec[0]
  return deltas, signs_deltas


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.

  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.

  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps


def compute_rdp_sample_without_replacement(q, noise_multiplier, steps, orders):
  """Compute RDP of Gaussian Mechanism using sampling without replacement.

  This function applies to the following schemes:
  1. Sampling w/o replacement: Sample a uniformly random subset of size m = q*n.
  2. ``Replace one data point'' version of differential privacy, i.e., n is
     considered public information.

  Reference: Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf (A strengthened
  version applies subsampled-Gaussian mechanism)
  - Wang, Balle, Kasiviswanathan. "Subsampled Renyi Differential Privacy and
  Analytical Moments Accountant." AISTATS'2019.

  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp_sample_without_replacement_scalar(
        q, noise_multiplier, orders)
  else:
    rdp = np.array([
        _compute_rdp_sample_without_replacement_scalar(q, noise_multiplier,
                                                       order)
        for order in orders
    ])

  return rdp * steps


def _compute_rdp_sample_without_replacement_scalar(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.

  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """

  assert (q <= 1) and (q >= 0) and (alpha >= 1)

  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  if float(alpha).is_integer():
    return _compute_rdp_sample_without_replacement_int(q, sigma, alpha) / (
        alpha - 1)
  else:
    # When alpha not an integer, we apply Corollary 10 of [WBK19] to interpolate
    # the CGF and obtain an upper bound
    alpha_f = math.floor(alpha)
    alpha_c = math.ceil(alpha)

    x = _compute_rdp_sample_without_replacement_int(q, sigma, alpha_f)
    y = _compute_rdp_sample_without_replacement_int(q, sigma, alpha_c)
    t = alpha - alpha_f
    return ((1 - t) * x + t * y) / (alpha - 1)


def _compute_rdp_sample_without_replacement_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha, subsampling without replacement.

  When alpha is smaller than max_alpha, compute the bound Theorem 27 exactly,
    otherwise compute the bound with Stirling approximation.

  Args:
    q: The sampling proportion = m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """

  max_alpha = 256
  assert isinstance(alpha, six.integer_types)

  if np.isinf(alpha):
    return np.inf
  elif alpha == 1:
    return 0

  def cgf(x):
    # Return rdp(x+1)*x, the rdp of Gaussian mechanism is alpha/(2*sigma**2)
    return x * 1.0 * (x + 1) / (2.0 * sigma**2)

  def func(x):
    # Return the rdp of Gaussian mechanism
    return 1.0 * x / (2.0 * sigma**2)

  # Initialize with 1 in the log space.
  log_a = 0
  # Calculates the log term when alpha = 2
  log_f2m1 = func(2.0) + np.log(1 - np.exp(-func(2.0)))
  if alpha <= max_alpha:
    # We need forward differences of exp(cgf)
    # The following line is the numerically stable way of implementing it.
    # The output is in polar form with logarithmic magnitude
    deltas, _ = _get_forward_diffs(cgf, alpha)
    # Compute the bound exactly requires book keeping of O(alpha**2)

    for i in range(2, alpha + 1):
      if i == 2:
        s = 2 * np.log(q) + _log_comb(alpha, 2) + np.minimum(
            np.log(4) + log_f2m1,
            func(2.0) + np.log(2))
      elif i > 2:
        delta_lo = deltas[int(2 * np.floor(i / 2.0)) - 1]
        delta_hi = deltas[int(2 * np.ceil(i / 2.0)) - 1]
        s = np.log(4) + 0.5 * (delta_lo + delta_hi)
        s = np.minimum(s, np.log(2) + cgf(i - 1))
        s += i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)
    return float(log_a)
  else:
    # Compute the bound with stirling approximation. Everything is O(x) now.
    for i in range(2, alpha + 1):
      if i == 2:
        s = 2 * np.log(q) + _log_comb(alpha, 2) + np.minimum(
            np.log(4) + log_f2m1,
            func(2.0) + np.log(2))
      else:
        s = np.log(2) + cgf(i - 1) + i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)

    return log_a


def compute_heterogenous_rdp(sampling_probabilities, noise_multipliers,
                             steps_list, orders):
  """Computes RDP of Heteregoneous Applications of Sampled Gaussian Mechanisms.

  Args:
    sampling_probabilities: A list containing the sampling rates.
    noise_multipliers: A list containing the noise multipliers: the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
    steps_list: A list containing the number of steps at each
      `sampling_probability` and `noise_multiplier`.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  assert len(sampling_probabilities) == len(noise_multipliers)

  rdp = 0
  for q, noise_multiplier, steps in zip(sampling_probabilities,
                                        noise_multipliers, steps_list):
    rdp += compute_rdp(q, noise_multiplier, steps, orders)

  return rdp


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Computes delta (or eps) for given eps (or delta) from RDP values.

  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta`
      must be `None`.

  Returns:
    A tuple of epsilon, delta, and the optimal order.

  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order


def compute_rdp_from_ledger(ledger, orders):
  """Computes RDP of Sampled Gaussian Mechanism from ledger.

  Args:
    ledger: A formatted privacy ledger.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    RDP at all orders. Can be `np.inf`.
  """
  total_rdp = np.zeros_like(orders, dtype=float)
  for sample in ledger:
    # Compute equivalent z from l2_clip_bounds and noise stddevs in sample.
    # See https://arxiv.org/pdf/1812.06210.pdf for derivation of this formula.
    effective_z = sum([
        (q.noise_stddev / q.l2_norm_bound)**-2 for q in sample.queries])**-0.5
    total_rdp += compute_rdp(
        sample.selection_probability, effective_z, 1, orders)
  return total_rdp

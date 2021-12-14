# Copyright 2021, The TensorFlow Authors.
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
"""Tests for discrete_gaussian_utils."""

import collections
import fractions
import math
import random

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import discrete_gaussian_utils

EXACT_SAMPLER_SEED = 4242


class DiscreteGaussianUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(dtype=[tf.bool, tf.float32, tf.float64])
  def test_raise_on_bad_dtype(self, dtype):
    with self.assertRaises(ValueError):
      _ = discrete_gaussian_utils.sample_discrete_gaussian(1, (1,), dtype)

  def test_raise_on_negative_scale(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = discrete_gaussian_utils.sample_discrete_gaussian(-10, (1,))

  def test_raise_on_float_scale(self):
    with self.assertRaises(TypeError):
      _ = discrete_gaussian_utils.sample_discrete_gaussian(3.14, (1,))

  @parameterized.product(shape=[(), (1,), (100,), (2, 2), (3, 3, 3),
                                (4, 1, 1, 1)])
  def test_shapes(self, shape):
    samples = discrete_gaussian_utils.sample_discrete_gaussian(100, shape)
    samples = self.evaluate(samples)
    self.assertAllEqual(samples.shape, shape)

  @parameterized.product(dtype=[tf.int32, tf.int64])
  def test_dtypes(self, dtype):
    samples = discrete_gaussian_utils.sample_discrete_gaussian(1, (10,), dtype)
    samples = self.evaluate(samples)
    # Convert output np dtypes to tf dtypes.
    self.assertEqual(tf.as_dtype(samples.dtype), dtype)

  def test_zero_noise(self):
    scale = 0
    shape = (100,)
    dtype = tf.int32
    samples = discrete_gaussian_utils.sample_discrete_gaussian(
        scale, shape, dtype=dtype)
    samples = self.evaluate(samples)
    self.assertAllEqual(samples, tf.zeros(shape, dtype=dtype))

  @parameterized.named_parameters([('small_scale_small_n', 10, 2000, 1, 2),
                                   ('small_scale_large_n', 10, 5000, 1, 1),
                                   ('large_scale_small_n', 50, 2000, 2, 5),
                                   ('large_scale_large_n', 50, 5000, 2, 3)])
  def test_match_exact_sampler(self, scale, num_samples, mean_std_atol,
                               percentile_atol):
    true_samples = exact_sampler(scale, num_samples)
    drawn_samples = discrete_gaussian_utils.sample_discrete_gaussian(
        scale=scale, shape=(num_samples,))
    drawn_samples = self.evaluate(drawn_samples)

    # Check mean, std, and percentiles.
    self.assertAllClose(
        np.mean(true_samples), np.mean(drawn_samples), atol=mean_std_atol)
    self.assertAllClose(
        np.std(true_samples), np.std(drawn_samples), atol=mean_std_atol)
    self.assertAllClose(
        np.percentile(true_samples, [10, 30, 50, 70, 90]),
        np.percentile(drawn_samples, [10, 30, 50, 70, 90]),
        atol=percentile_atol)

  @parameterized.named_parameters([('n_1000', 1000, 5e-2),
                                   ('n_10000', 10000, 5e-3)])
  def test_kl_divergence(self, num_samples, kl_tolerance):
    """Compute KL divergence betwen empirical & true distribution."""
    scale = 10
    sq_sigma = scale * scale
    drawn_samples = discrete_gaussian_utils.sample_discrete_gaussian(
        scale=scale, shape=(num_samples,))
    drawn_samples = self.evaluate(drawn_samples)
    value_counts = collections.Counter(drawn_samples)

    kl = 0
    norm_const = dgauss_normalizing_constant(sq_sigma)

    for value, count in value_counts.items():
      kl += count * (
          math.log(count * norm_const / num_samples) + value * value /
          (2.0 * sq_sigma))

    kl /= num_samples
    self.assertLess(kl, kl_tolerance)


def exact_sampler(scale, num_samples, seed=EXACT_SAMPLER_SEED):
  """Implementation of the exact discrete gaussian distribution sampler.

  Source: https://arxiv.org/pdf/2004.00010.pdf.

  Args:
    scale: The scale of the discrete Gaussian.
    num_samples: The number of samples to generate.
    seed: The seed for the random number generator to reproduce samples.

  Returns:
    A numpy array of discrete Gaussian samples.
  """

  def randrange(a, rng):
    return rng.randrange(a)

  def bern_em1(rng):
    """Sample from Bernoulli(exp(-1))."""
    k = 2
    while True:
      if randrange(k, rng) == 0:  # if Bernoulli(1/k)==1
        k = k + 1
      else:
        return k % 2

  def bern_emab1(a, b, rng):
    """Sample from Bernoulli(exp(-a/b)), assuming 0 <= a <= b."""
    assert isinstance(a, int)
    assert isinstance(b, int)
    assert 0 <= a <= b
    k = 1
    while True:
      if randrange(b, rng) < a and randrange(k, rng) == 0:  # if Bern(a/b/k)==1
        k = k + 1
      else:
        return k % 2

  def bern_emab(a, b, rng):
    """Sample from Bernoulli(exp(-a/b)), allowing a > b."""
    while a > b:
      if bern_em1(rng) == 0:
        return 0
      a = a - b
    return bern_emab1(a, b, rng)

  def geometric(t, rng):
    """Sample from geometric(1-exp(-1/t))."""
    assert isinstance(t, int)
    assert t > 0
    while True:
      u = randrange(t, rng)
      if bern_emab1(u, t, rng) == 1:
        while bern_em1(rng) == 1:
          u = u + t
        return u

  def dlap(t, rng):
    """Sample from discrete Laplace with scale t.

    Pr[x] = exp(-|x|/t) * (exp(1/t)-1)/(exp(1/t)+1). Supported on integers.

    Args:
      t: The scale.
      rng: The random number generator.

    Returns:
      A discrete Laplace sample.
    """
    assert isinstance(t, int)
    assert t > 0
    while True:
      u = geometric(t, rng)
      b = randrange(2, rng)
      if b == 1:
        return u
      elif u > 0:
        return -u

  def floorsqrt(x):
    """Compute floor(sqrt(x)) exactly."""
    assert x >= 0
    a = 0  # maintain a^2<=x.
    b = 1  # maintain b^2>x.
    while b * b <= x:
      b = 2 * b
    # Do binary search.
    while a + 1 < b:
      c = (a + b) // 2
      if c * c <= x:
        a = c
      else:
        b = c
    return a

  def dgauss(ss, num, rng):
    """Sample from discrete Gaussian.

    Args:
      ss: Variance proxy, squared scale, sigma^2.
      num: The number of samples to generate.
      rng: The random number generator.

    Returns:
      A list of discrete Gaussian samples.
    """
    ss = fractions.Fraction(ss)  # cast to rational for exact arithmetic
    assert ss > 0
    t = floorsqrt(ss) + 1
    results = []
    trials = 0
    while len(results) < num:
      trials = trials + 1
      y = dlap(t, rng)
      z = (abs(y) - ss / t)**2 / (2 * ss)
      if bern_emab(z.numerator, z.denominator, rng) == 1:
        results.append(y)
    return results, t, trials

  rng = random.Random(seed)
  return np.array(dgauss(scale * scale, num_samples, rng)[0])


def dgauss_normalizing_constant(sigma_sq):
  """Compute the normalizing constant of the discrete Gaussian.

  Source: https://arxiv.org/pdf/2004.00010.pdf.

  Args:
    sigma_sq: Variance proxy, squared scale, sigma^2.

  Returns:
    The normalizing constant.
  """
  original = None
  poisson = None
  if sigma_sq <= 1:
    original = 0
    x = 1000
    while x > 0:
      original = original + math.exp(-x * x / (2.0 * sigma_sq))
      x = x - 1
    original = 2 * original + 1

  if sigma_sq * 100 >= 1:
    poisson = 0
    y = 1000
    while y > 0:
      poisson = poisson + math.exp(-math.pi * math.pi * sigma_sq * 2 * y * y)
      y = y - 1
    poisson = math.sqrt(2 * math.pi * sigma_sq) * (1 + 2 * poisson)

  if poisson is None:
    return original
  if original is None:
    return poisson

  scale = max(1, math.sqrt(2 * math.pi * sigma_sq))
  precision = 1e-15
  assert -precision * scale <= original - poisson <= precision * scale
  return (original + poisson) / 2


if __name__ == '__main__':
  tf.test.main()

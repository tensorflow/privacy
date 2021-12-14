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
"""Util functions for drawing discrete Gaussian samples.

The following functions implement a vectorized TF version of the sampling
algorithm described in the paper:

The Discrete Gaussian for Differential Privacy
https://arxiv.org/pdf/2004.00010.pdf

Note that the exact sampling implementation should use integer and fractional
parameters only. Here, we relax this constraint a bit and use vectorized
implementations of Bernoulli and discrete Laplace sampling that can take float
parameters.
"""

import tensorflow as tf
import tensorflow_probability as tf_prob


def _sample_discrete_laplace(t, shape):
  """Sample from discrete Laplace with scale t.

  This method is based on the observation that sampling from Z ~ Lap(t) is
  equivalent to sampling X, Y independently from Geo(1 - exp(-1/t)) and take
  Z = X - Y.

  Note also that tensorflow_probability's geometric sampler is based on floating
  operations and may possibly be inexact.

  Args:
    t: The scale of the discrete Laplace distribution.
    shape: The tensor shape of the tensors drawn.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  geometric_probs = 1.0 - tf.exp(-1.0 / tf.cast(t, tf.float64))
  sampler = tf_prob.distributions.Geometric(probs=geometric_probs)
  return tf.cast(sampler.sample(shape) - sampler.sample(shape), tf.int64)


def _sample_bernoulli(p):
  """Sample from Bernoulli(p)."""
  return tf_prob.distributions.Bernoulli(probs=p, dtype=tf.int64).sample()


def _check_input_args(scale, shape, dtype):
  """Checks the input args to the discrete Gaussian sampler."""
  if tf.as_dtype(dtype) not in (tf.int32, tf.int64):
    raise ValueError(
        f'Only tf.int32 and tf.int64 are supported. Found dtype `{dtype}`.')

  checks = [
      tf.compat.v1.assert_non_negative(scale),
      tf.compat.v1.assert_integer(scale)
  ]
  with tf.control_dependencies(checks):
    return tf.identity(scale), shape, dtype


def _int_square(value):
  """Avoids the TF op `Square(T=...)` for ints as sampling can happen on clients."""
  return (value - 1) * (value + 1) + 1


@tf.function
def _sample_discrete_gaussian_helper(scale, shape, dtype):
  """Draw samples from discrete Gaussian, assuming scale >= 0."""
  scale = tf.cast(scale, tf.int64)
  sq_scale = _int_square(scale)

  # Scale for discrete Laplace. The sampling algorithm should be correct
  # for any discrete Laplace scale, and the original paper uses
  # `dlap_scale = floor(scale) + 1`. Here we use `dlap_scale = scale` (where
  # input `scale` is restricted to integers >= 1) to simplify the fraction
  # below. It turns out that for integer scales >= 1, `dlap_scale = scale` gives
  # a good minimum success rate of ~70%, allowing a small oversampling factor.
  dlap_scale = scale
  oversample_factor = 1.5

  # Draw at least some samples in case we got unlucky with small input shape.
  min_n = 1000
  target_n = tf.reduce_prod(tf.cast(shape, tf.int64))
  oversample_n = oversample_factor * tf.cast(target_n, tf.float32)
  draw_n = tf.maximum(min_n, tf.cast(oversample_n, tf.int32))

  accepted_n = tf.constant(0, dtype=target_n.dtype)
  result = tf.zeros((0,), dtype=tf.int64)

  while accepted_n < target_n:
    # Since the number of samples could be different in every retry, we need to
    # manually specify the shape info for TF.
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(result, tf.TensorShape([None]))])
    # Draw samples.
    samples = _sample_discrete_laplace(dlap_scale, shape=(draw_n,))
    z_numer = _int_square(tf.abs(samples) - scale)
    z_denom = 2 * sq_scale
    bern_probs = tf.exp(-1.0 * tf.divide(z_numer, z_denom))
    accept = _sample_bernoulli(bern_probs)
    # Keep successful samples and increment counter.
    accepted_samples = samples[tf.equal(accept, 1)]
    accepted_n += tf.cast(tf.size(accepted_samples), accepted_n.dtype)
    result = tf.concat([result, accepted_samples], axis=0)
    # Reduce the number of draws for any retries.
    draw_n = tf.cast(target_n - accepted_n, tf.float32) * oversample_factor
    draw_n = tf.maximum(min_n, tf.cast(draw_n, tf.int32))

  return tf.cast(tf.reshape(result[:target_n], shape), dtype)


def sample_discrete_gaussian(scale, shape, dtype=tf.int32):
  """Draws (possibly inexact) samples from the discrete Gaussian distribution.

  We relax some integer constraints to use vectorized implementations of
  Bernoulli and discrete Laplace sampling. Integer operations are done in
  tf.int64 as TF does not have direct support for fractions.

  Args:
    scale: The scale of the discrete Gaussian distribution.
    shape: The shape of the output tensor.
    dtype: The type of the output.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  scale, shape, dtype = _check_input_args(scale, shape, dtype)
  return tf.cond(
      tf.equal(scale, 0), lambda: tf.zeros(shape, dtype),
      lambda: _sample_discrete_gaussian_helper(scale, shape, dtype))

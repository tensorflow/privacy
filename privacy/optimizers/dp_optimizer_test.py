# Copyright 2018, The TensorFlow Authors.
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

"""Tests for differentially private optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.optimizers import dp_adam
from tensorflow_privacy.privacy.optimizers import dp_gradient_descent


def loss(val0, val1):
  """Loss function that is minimized at the mean of the input points."""
  return 0.5 * tf.reduce_sum(tf.squared_difference(val0, val1), axis=1)


class DPOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  # Parameters for testing: optimizer, nb_microbatches, expected answer.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_gradient_descent.DPGradientDescentOptimizer, 1,
       [-10.0, -10.0]),
      ('DPGradientDescent 2', dp_gradient_descent.DPGradientDescentOptimizer, 2,
       [-5.0, -5.0]),
      ('DPGradientDescent 4', dp_gradient_descent.DPGradientDescentOptimizer, 4,
       [-2.5, -2.5]), ('DPAdam 1', dp_adam.DPAdamOptimizer, 1, [-10.0, -10.0]),
      ('DPAdam 2', dp_adam.DPAdamOptimizer, 2, [-5.0, -5.0]),
      ('DPAdam 4', dp_adam.DPAdamOptimizer, 4, [-2.5, -2.5]))
  def testBaseline(self, cls, nb_microbatches, expected_answer):
    with self.cached_session() as sess:
      var0 = tf.Variable([1.0, 2.0])
      data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])

      opt = cls(learning_rate=2.0, nb_microbatches=nb_microbatches)
      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))

      # Expected gradient is sum of differences divided by number of
      # microbatches.
      gradient_op = opt.compute_gradients(loss(data0, var0), [var0])
      grads_and_vars = sess.run(gradient_op)
      self.assertAllCloseAccordingToType(expected_answer, grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_gradient_descent.DPGradientDescentOptimizer),
      ('DPAdam', dp_adam.DPAdamOptimizer))
  def testClippingNorm(self, cls):
    with self.cached_session() as sess:
      var0 = tf.Variable([0.0, 0.0])
      data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])

      opt = cls(learning_rate=2.0, l2_norm_clip=1.0, nb_microbatches=1)
      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([0.0, 0.0], self.evaluate(var0))

      # Expected gradient is sum of differences.
      gradient_op = opt.compute_gradients(loss(data0, var0), [var0])
      grads_and_vars = sess.run(gradient_op)
      self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_gradient_descent.DPGradientDescentOptimizer),
      ('DPAdam', dp_adam.DPAdamOptimizer))
  def testNoiseMultiplier(self, cls):
    with self.cached_session() as sess:
      var0 = tf.Variable([0.0])
      data0 = tf.Variable([[0.0]])

      opt = cls(
          learning_rate=2.0,
          l2_norm_clip=4.0,
          noise_multiplier=2.0,
          nb_microbatches=1)
      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([0.0], self.evaluate(var0))

      gradient_op = opt.compute_gradients(loss(data0, var0), [var0])
      grads = []
      for _ in xrange(1000):
        grads_and_vars = sess.run(gradient_op)
        grads.append(grads_and_vars[0][0])

      # Test standard deviation is close to l2_norm_clip * noise_multiplier.
      self.assertNear(np.std(grads), 2.0 * 4.0, 0.5)


if __name__ == '__main__':
  tf.test.main()

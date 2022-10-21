# Copyright 2022, The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dp_optimizer_keras_sparse.

The tests here are branched from dp_optimizer_keras_test.py with some
extra tests, specifically `testSparseTensor`, `testNoiseMultiplier`, and
`testNoGetGradients`, for testing the difference between
dp_optimizer_keras_sparse and dp_optimizer_keras, as outlined in the
docstring of make_sparse_keras_optimizer_class.
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_sparse as dp_optimizer


class DPOptimizerTest(tf.test.TestCase, parameterized.TestCase):
  """Tests dp_optimizer_keras_sparse optimizers."""

  def _loss(self, val0, val1):
    """Loss function whose derivative w.r.t val1 is val1 - val0."""
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer.DPSparseKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer.DPSparseKerasAdamOptimizer, 2,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagrad 4', dp_optimizer.DPSparseKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
  )
  def testBaselineWithCallableLoss(self, cls, num_microbatches, expected_grad0,
                                   expected_grad1):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = opt._compute_gradients(loss, [var0, var1])
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer.DPSparseKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer.DPSparseKerasAdamOptimizer, 2, [-2.5, -2.5],
       [-0.5]),
      ('DPAdagrad 4', dp_optimizer.DPSparseKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
  )
  def testBaselineWithTensorLoss(self, cls, num_microbatches, expected_grad0,
                                 expected_grad1):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    tape = tf.GradientTape()
    with tape:
      loss = self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = opt._compute_gradients(loss, [var0, var1], tape=tape)
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer.DPSparseKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer.DPSparseKerasAdamOptimizer, 2, [-2.5, -2.5],
       [-0.5]),
      ('DPAdagrad 4', dp_optimizer.DPSparseKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
  )
  def testSparseTensor(self, cls, num_microbatches, expected_grad0,
                       expected_grad1):
    # Keep all the tensors to its sparse form
    dp_optimizer._KEEP_SPARSE_THRESHOLD = 0
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    def loss():
      return (self._loss(data0, tf.gather(var0, tf.constant([0, 1]))) +
              self._loss(data1, var1))

    grads_and_vars = opt._compute_gradients(loss, [var0, var1])
    self.assertIsInstance(grads_and_vars[0][0], tf.IndexedSlices)
    self.assertAllCloseAccordingToType(
        expected_grad0, tf.convert_to_tensor(grads_and_vars[0][0]))
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer.DPSparseKerasSGDOptimizer),
  )
  def testClippingNorm(self, cls):
    var0 = tf.Variable([0.0, 0.0])
    data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])

    opt = cls(
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    # Expected gradient is sum of differences.
    grads_and_vars = opt._compute_gradients(loss, [var0])
    self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer.DPSparseKerasSGDOptimizer, 2.5, 1),
      ('DPGradientDescent 2', dp_optimizer.DPSparseKerasSGDOptimizer, 2.5, 2),
      ('DPGradientDescent 4', dp_optimizer.DPSparseKerasSGDOptimizer, 2.5, 4),
  )
  def testClippingNormMultipleVariables(self, cls, l2_norm_clip,
                                        num_microbatches):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 6.0], [5.0, 6.0], [4.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    # Expected gradient is sum of differences.
    grads_and_vars = opt._compute_gradients(loss, [var0, var1])

    # Compute expected gradients.
    batch_size = data0.shape[0]
    grad0 = (data0 - var0).numpy()
    grad1 = (data1 - var1).numpy()
    grads = np.concatenate([grad0, grad1], axis=1)

    grads = np.reshape(
        grads, (num_microbatches, int(batch_size / num_microbatches), -1))
    grads = np.mean(grads, axis=1)

    norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=grads)
    grad_factors = l2_norm_clip / np.maximum(l2_norm_clip, norms)

    scaled_grads = grads * grad_factors[:, None]
    mean_scaled_grads = -np.mean(scaled_grads, axis=0)
    expected0, expected1 = np.split(mean_scaled_grads, [2], axis=0)

    # Compare expected with actual gradients.
    self.assertAllCloseAccordingToType(expected0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer.DPSparseKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer.DPSparseKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer.DPSparseKerasAdamOptimizer),
  )
  def testAssertOnNoCallOfComputeGradients(self, cls):
    """Tests that assertion fails when DP gradients are not computed."""
    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    with self.assertRaises(AssertionError):
      grads_and_vars = tf.Variable([0.0])
      opt.apply_gradients(grads_and_vars)

    # Expect no exception if _compute_gradients is called.
    var0 = tf.Variable([0.0])
    data0 = tf.Variable([[0.0]])
    loss = lambda: self._loss(data0, var0)
    grads_and_vars = opt._compute_gradients(loss, [var0])
    opt.apply_gradients(grads_and_vars)

  # Parameters for testing: optimizer, l2_norm_clip, noise_multiplier,
  # num_microbatches, gradient_accumulation_steps
  @parameterized.named_parameters(
      ('DPGradientDescent Dense 2 4 1 1',
       dp_optimizer.DPSparseKerasSGDOptimizer, False, 2.0, 4.0, 1, 1),
      ('DPGradientDescent Dense 3 2 4 2',
       dp_optimizer.DPSparseKerasSGDOptimizer, False, 3.0, 2.0, 4, 2),
      ('DPGradientDescent Dense 8 6 8 3',
       dp_optimizer.DPSparseKerasSGDOptimizer, False, 8.0, 6.0, 8, 3),
      ('DPGradientDescent Dense 8 6 None 3',
       dp_optimizer.DPSparseKerasSGDOptimizer, False, 8.0, 6.0, None, 3),
      ('DPGradientDescent Sparse 2 4 1 1',
       dp_optimizer.DPSparseKerasSGDOptimizer, True, 2.0, 4.0, 1, 1),
      ('DPGradientDescent Sparse 3 2 4 2',
       dp_optimizer.DPSparseKerasSGDOptimizer, True, 3.0, 2.0, 4, 2),
      ('DPGradientDescent Sparse 8 6 8 3',
       dp_optimizer.DPSparseKerasSGDOptimizer, True, 8.0, 6.0, 8, 3),
      ('DPGradientDescent Sparse 8 6 None 3',
       dp_optimizer.DPSparseKerasSGDOptimizer, True, 8.0, 6.0, None, 3),
  )
  def testNoiseMultiplier(
      self, cls, use_embeddings, l2_norm_clip, noise_multiplier,
      num_microbatches, gradient_accumulation_steps):
    """Tests that DP optimizer works with keras optimizer."""
    dp_optimizer._KEEP_SPARSE_THRESHOLD = 0
    inputs = {'x': tf.keras.Input(shape=(1000)),
              'i': tf.keras.Input(shape=(1))}
    if use_embeddings:
      # Emulates a linear layer using embeddings.
      layer = tf.keras.layers.Embedding(
          10,
          1000,
          embeddings_initializer='zeros')
      preds = tf.reduce_sum(
          tf.multiply(layer(inputs['i']), inputs['x']), axis=1, keepdims=True)
      weights = layer.embeddings
    else:
      layer = tf.keras.layers.Dense(
          1,
          activation='linear',
          name='dense',
          kernel_initializer='zeros',
          bias_initializer='zeros')
      preds = layer(inputs['x'])
      weights = layer.kernel
    model = tf.keras.Model(inputs, preds)

    loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    optimizer = cls(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1.0)

    model.compile(optimizer=optimizer, loss=loss)

    true_weights = np.zeros((1000, 1), dtype=np.float32)
    true_bias = np.array([0.0]).astype(np.float32)
    for _ in range(9 * gradient_accumulation_steps):
      x = np.zeros((16, 1000), dtype=np.float32)
      i = np.random.randint(2, size=(16, 1))
      y = np.matmul(x, true_weights) + true_bias
      model.fit(x={'x': x, 'i': i}, y=y)

    if num_microbatches is None:
      num_microbatches = 16
    noise_stddev = (3 * l2_norm_clip * noise_multiplier / num_microbatches /
                    gradient_accumulation_steps)
    self.assertNear(np.std(weights), noise_stddev, 0.5)

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer.DPSparseKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer.DPSparseKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer.DPSparseKerasAdamOptimizer),
  )
  def testNoGetGradients(self, cls):
    """Tests that get_gradients raises an error."""
    var0 = tf.Variable([[1.0, 2.0]], dtype=tf.float32)
    x1 = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    loss1 = lambda: tf.matmul(var0, x1, transpose_b=True)
    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    with self.assertRaises(ValueError):
      opt.get_gradients(loss1, var0)

  def testLargeBatchEmulationNoNoise(self):
    # Test for emulation of large batch training.
    # It tests that updates are only done every gradient_accumulation_steps
    # steps.
    # In this test we set noise multiplier to zero and clipping norm to high
    # value, such that optimizer essentially behave as non-DP optimizer.
    # This makes easier to check how values of variables are changing.
    #
    # This test optimizes loss var0*x + var1
    # Gradients of this loss are computed as:
    # d(loss)/d(var0) = x
    # d(loss)/d(var1) = 1
    var0 = tf.Variable([[1.0, 2.0]], dtype=tf.float32)
    var1 = tf.Variable([3.0], dtype=tf.float32)
    x1 = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    loss1 = lambda: tf.matmul(var0, x1, transpose_b=True) + var1
    x2 = tf.constant([[4.0, 2.0], [2.0, 1.0]], dtype=tf.float32)
    loss2 = lambda: tf.matmul(var0, x2, transpose_b=True) + var1

    opt = dp_optimizer.DPSparseKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        gradient_accumulation_steps=2,
        learning_rate=1.0)

    # before any call to optimizer
    self.assertAllCloseAccordingToType([[1.0, 2.0]], var0)
    self.assertAllCloseAccordingToType([3.0], var1)

    opt.minimize(loss1, [var0, var1])
    # After first call to optimizer values didn't change
    self.assertAllCloseAccordingToType([[1.0, 2.0]], var0)
    self.assertAllCloseAccordingToType([3.0], var1)

    opt.minimize(loss2, [var0, var1])
    # After second call to optimizer updates were applied
    self.assertAllCloseAccordingToType([[-1.0, 1.0]], var0)
    self.assertAllCloseAccordingToType([2.0], var1)

    opt.minimize(loss2, [var0, var1])
    # After third call to optimizer values didn't change
    self.assertAllCloseAccordingToType([[-1.0, 1.0]], var0)
    self.assertAllCloseAccordingToType([2.0], var1)

    opt.minimize(loss2, [var0, var1])
    # After fourth call to optimizer updates were applied again
    self.assertAllCloseAccordingToType([[-4.0, -0.5]], var0)
    self.assertAllCloseAccordingToType([1.0], var1)

  @parameterized.named_parameters(
      ('DPSparseKerasSGDOptimizer 1',
       dp_optimizer.DPSparseKerasSGDOptimizer, 1),
      ('DPSparseKerasSGDOptimizer 2',
       dp_optimizer.DPSparseKerasSGDOptimizer, 2),
      ('DPSparseKerasSGDOptimizer 4',
       dp_optimizer.DPSparseKerasSGDOptimizer, 4),
      ('DPSparseKerasAdamOptimizer 2',
       dp_optimizer.DPSparseKerasAdamOptimizer, 1),
      ('DPSparseKerasAdagradOptimizer 2',
       dp_optimizer.DPSparseKerasAdagradOptimizer, 2),
  )
  def testLargeBatchEmulation(self, cls, gradient_accumulation_steps):
    # Tests various optimizers with large batch emulation.
    # Uses clipping and noise, thus does not test specific values
    # of the variables and only tests how often variables are updated.
    var0 = tf.Variable([[1.0, 2.0]], dtype=tf.float32)
    var1 = tf.Variable([3.0], dtype=tf.float32)
    x = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    loss = lambda: tf.matmul(var0, x, transpose_b=True) + var1

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1.0)

    for _ in range(gradient_accumulation_steps):
      self.assertAllCloseAccordingToType([[1.0, 2.0]], var0)
      self.assertAllCloseAccordingToType([3.0], var1)
      opt.minimize(loss, [var0, var1])

    self.assertNotAllClose([[1.0, 2.0]], var0)
    self.assertNotAllClose([3.0], var1)

  def testKerasModelBaselineSaving(self):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer.DPSparseKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=None,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

    tempdir = self.create_tempdir()
    model.save(tempdir, save_format='tf')

  def testKerasModelBaselineAfterSavingLoading(self):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer.DPSparseKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=None,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.predict(train_data, batch_size=8)
    tempdir = self.create_tempdir()
    model.save(tempdir, save_format='tf')
    model.load_weights(tempdir)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

  @parameterized.named_parameters(('1', 1), ('None', None))
  def testKerasModelBaselineNoNoise(self, num_microbatches):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer.DPSparseKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

    self.assertAllClose(model.get_weights()[0], true_weights, atol=0.05)
    self.assertAllClose(model.get_weights()[1], true_bias, atol=0.05)


if __name__ == '__main__':
  tf.test.main()

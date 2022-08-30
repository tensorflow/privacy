# Copyright 2019, The TensorFlow Authors.
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


from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_vectorized


class DPOptimizerComputeGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for _compute_gradients method."""

  def _loss(self, val0, val1):
    """Loss function whose derivative w.r.t val1 is val1 - val0."""
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer_keras.DPKerasAdamOptimizer, 2, [-2.5, -2.5
                                                               ], [-0.5]),
      ('DPAdagrad 4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
      ('DPGradientDescentVectorized 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdamVectorized 2',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer, 2,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagradVectorized 4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagradVectorized None',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, None,
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
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer_keras.DPKerasAdamOptimizer, 2, [-2.5, -2.5
                                                               ], [-0.5]),
      ('DPAdagrad 4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
      ('DPGradientDescentVectorized 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdamVectorized 2',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer, 2,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagradVectorized 4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagradVectorized None',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, None,
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

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),
      ('DPGradientDescentVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),
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
      ('DPGradientDescent 2 4 1', dp_optimizer_keras.DPKerasSGDOptimizer, 2.0,
       4.0, 1),
      ('DPGradientDescent 4 1 4', dp_optimizer_keras.DPKerasSGDOptimizer, 4.0,
       1.0, 4),
      ('DPGradientDescentVectorized 2 4 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 2.0, 4.0,
       1),
      ('DPGradientDescentVectorized 4 1 4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 4.0, 1.0,
       4),
  )
  def testNoiseMultiplier(self, cls, l2_norm_clip, noise_multiplier,
                          num_microbatches):
    var0 = tf.Variable(tf.zeros([1000], dtype=tf.float32))
    data0 = tf.Variable(tf.zeros([16, 1000], dtype=tf.float32))

    opt = cls(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    grads_and_vars = opt._compute_gradients(loss, [var0])
    grads = grads_and_vars[0][0].numpy()

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.
    self.assertNear(
        np.std(grads), l2_norm_clip * noise_multiplier / num_microbatches, 0.5)

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer_keras.DPKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer_keras.DPKerasAdamOptimizer),
      ('DPGradientDescentVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),
      ('DPAdagradVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer),
      ('DPAdamVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer),
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


class DPOptimizerGetGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for get_gradient method.

  Since get_gradients must run in graph mode, the method is tested within
  the Estimator framework.
  """

  def _make_linear_model_fn(self, opt_cls, l2_norm_clip, noise_multiplier,
                            num_microbatches, learning_rate):
    """Returns a model function for a linear regressor."""

    def linear_model_fn(features, labels, mode):
      layer = tf.keras.layers.Dense(
          1,
          activation='linear',
          name='dense',
          kernel_initializer='zeros',
          bias_initializer='zeros')
      preds = layer(features)

      vector_loss = 0.5 * tf.math.squared_difference(labels, preds)
      scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

      optimizer = opt_cls(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          num_microbatches=num_microbatches,
          learning_rate=learning_rate)

      params = layer.trainable_weights
      global_step = tf.compat.v1.train.get_global_step()
      train_op = tf.group(
          optimizer.get_updates(loss=vector_loss, params=params),
          [tf.compat.v1.assign_add(global_step, 1)])
      return tf_estimator.EstimatorSpec(
          mode=mode, loss=scalar_loss, train_op=train_op)

    return linear_model_fn

  # Parameters for testing: optimizer, num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPGradientDescent 2', dp_optimizer_keras.DPKerasSGDOptimizer, 2),
      ('DPGradientDescent 4', dp_optimizer_keras.DPKerasSGDOptimizer, 4),
      ('DPGradientDescentVectorized 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1),
      ('DPGradientDescentVectorized 2',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 2),
      ('DPGradientDescentVectorized 4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 4),
      ('DPGradientDescentVectorized None',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, None),
  )
  def testBaseline(self, cls, num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    linear_regressor = tf_estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 100.0, 0.0, num_microbatches,
                                            0.05))

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)

    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    linear_regressor.train(input_fn=train_input_fn, steps=125)

    self.assertAllClose(
        linear_regressor.get_variable_value('dense/kernel'),
        true_weights,
        atol=0.05)
    self.assertAllClose(
        linear_regressor.get_variable_value('dense/bias'), true_bias, atol=0.05)

  # Parameters for testing: optimizer, num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPGradientDescentVectorized 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1),
  )
  def testClippingNorm(self, cls, num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    true_weights = np.array([[6.0], [0.0], [0], [0]]).astype(np.float32)
    true_bias = np.array([0]).astype(np.float32)

    train_data = np.array([[1.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    train_labels = np.matmul(train_data, true_weights) + true_bias

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(1)

    unclipped_linear_regressor = tf_estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 1.0e9, 0.0, num_microbatches,
                                            1.0))
    unclipped_linear_regressor.train(input_fn=train_input_fn, steps=1)

    kernel_value = unclipped_linear_regressor.get_variable_value('dense/kernel')
    bias_value = unclipped_linear_regressor.get_variable_value('dense/bias')
    global_norm = np.linalg.norm(np.concatenate((kernel_value, [bias_value])))

    clipped_linear_regressor = tf_estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 1.0, 0.0, num_microbatches,
                                            1.0))
    clipped_linear_regressor.train(input_fn=train_input_fn, steps=1)

    self.assertAllClose(
        clipped_linear_regressor.get_variable_value('dense/kernel'),
        kernel_value / global_norm,
        atol=0.001)
    self.assertAllClose(
        clipped_linear_regressor.get_variable_value('dense/bias'),
        bias_value / global_norm,
        atol=0.001)

  # Parameters for testing: optimizer, l2_norm_clip, noise_multiplier,
  # num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 2 4 1', dp_optimizer_keras.DPKerasSGDOptimizer, 2.0,
       4.0, 1),
      ('DPGradientDescent 3 2 4', dp_optimizer_keras.DPKerasSGDOptimizer, 3.0,
       2.0, 4),
      ('DPGradientDescent 8 6 8', dp_optimizer_keras.DPKerasSGDOptimizer, 8.0,
       6.0, 8),
      ('DPGradientDescentVectorized 2 4 1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 2.0, 4.0,
       1),
      ('DPGradientDescentVectorized 3 2 4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 3.0, 2.0,
       4),
      ('DPGradientDescentVectorized 8 6 8',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 8.0, 6.0,
       8),
  )
  def testNoiseMultiplier(self, cls, l2_norm_clip, noise_multiplier,
                          num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    linear_regressor = tf_estimator.Estimator(
        model_fn=self._make_linear_model_fn(
            cls,
            l2_norm_clip,
            noise_multiplier,
            num_microbatches,
            learning_rate=1.0))

    true_weights = np.zeros((1000, 1), dtype=np.float32)
    true_bias = np.array([0.0]).astype(np.float32)

    train_data = np.zeros((16, 1000), dtype=np.float32)
    train_labels = np.matmul(train_data, true_weights) + true_bias

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(16)

    linear_regressor.train(input_fn=train_input_fn, steps=1)

    kernel_value = linear_regressor.get_variable_value('dense/kernel')
    self.assertNear(
        np.std(kernel_value),
        l2_norm_clip * noise_multiplier / num_microbatches, 0.5)

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer_keras.DPKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer_keras.DPKerasAdamOptimizer),
      ('DPGradientDescentVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),
      ('DPAdagradVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer),
      ('DPAdamVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer),
  )
  def testAssertOnNoCallOfGetGradients(self, cls):
    """Tests that assertion fails when DP gradients are not computed."""
    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    with self.assertRaises(AssertionError):
      grads_and_vars = tf.Variable([0.0])
      opt.apply_gradients(grads_and_vars)

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

    opt = dp_optimizer_keras.DPKerasSGDOptimizer(
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
      ('DPKerasSGDOptimizer 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPKerasSGDOptimizer 2', dp_optimizer_keras.DPKerasSGDOptimizer, 2),
      ('DPKerasSGDOptimizer 4', dp_optimizer_keras.DPKerasSGDOptimizer, 4),
      ('DPKerasAdamOptimizer 2', dp_optimizer_keras.DPKerasAdamOptimizer, 1),
      ('DPKerasAdagradOptimizer 2', dp_optimizer_keras.DPKerasAdagradOptimizer,
       2),
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


class SimpleEmbeddingModel(tf.keras.Model):
  """Simple embedding model."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.embed_layer = tf.keras.layers.Embedding(
        name='embedding',
        input_dim=10,  # vocabulary size.
        output_dim=6,  # embedding size.
        embeddings_initializer='uniform',
        input_length=4)  # sequence length.
    self.pool_layer = tf.keras.layers.Dense(
        name='pooler',
        units=6,
        activation='tanh',
        kernel_initializer='zeros',
        bias_initializer='zeros')
    self.probs_layer = tf.keras.layers.Dense(
        units=1, activation='softmax', name='classification')

  def call(self, inputs, training=None):
    # The shape of the sequence output from the embedding layer is
    # [batch_size, sequence_length, embedding_size]
    sequence_output = self.embed_layer(inputs)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    # The shape of the pooled output from the embedding layer is
    # [batch_size, embedding_size]
    pooled_output = self.pool_layer(first_token_tensor)
    return sequence_output, pooled_output


def keras_embedding_model_fn(opt_cls,
                             l2_norm_clip: float,
                             noise_multiplier: float,
                             num_microbatches: int,
                             learning_rate: float,
                             use_seq_output: bool = False,
                             unconnected_gradients_to_zero: bool = False):
  """Construct a simple embedding model with a classification layer."""

  # Every sample has 4 tokens (sequence length=4).
  x = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name='input')
  sequence_output, pooled_output = SimpleEmbeddingModel()(x)
  if use_seq_output:
    embedding = sequence_output
  else:
    embedding = pooled_output
  probs = tf.keras.layers.Dense(
      units=1, activation='softmax', name='classification')(
          embedding)
  model = tf.keras.Model(inputs=x, outputs=probs, name='model')

  optimizer = opt_cls(
      l2_norm_clip=l2_norm_clip,
      noise_multiplier=noise_multiplier,
      num_microbatches=num_microbatches,
      unconnected_gradients_to_zero=unconnected_gradients_to_zero,
      learning_rate=learning_rate)

  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.MeanSquaredError(
          # Return per-sample loss
          reduction=tf.keras.losses.Reduction.NONE),
      metrics=['accuracy'])
  return model


class DPVectorizedOptimizerUnconnectedNodesTest(tf.test.TestCase,
                                                parameterized.TestCase):
  """Tests for vectorized optimizers when there are unconnected nodes.

  Subclassed Keras models can have layers that are defined in the graph, but
  not connected to the input or output. Or a condition expression could
  determine if the layer in question was connected or not. In such cases, the
  gradients are not present for that unconnected layer. The vectorized DP
  optimizers compute the per-microbatch losses using the Jacobian. The Jacobian
  will contain 'None' values corresponding to that layer. This causes an error
  in the gradient computation.
  This error can be mitigated by setting those unconnected gradients to 0
  instead of 'None'. This is done using the 'unconnected_gradients' flag of the
  tf.GradientTape.jacobian() method.
  This class of tests tests the possible combinations of presence/absence of
  unconnected layers and setting unconnected gradients to 'None' or 0. In these
  tests, this is done by setting 'unconnected_gradients_to_zero' to True if the
  gradients are to be set to zero, or False if they are to be set to None.
  """

  # Parameters for testing: optimizer.
  @parameterized.named_parameters(
      ('DPSGDVectorized_SeqOutput_UnconnectedGradients',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),)
  def testSeqOutputUnconnectedGradientsAsNoneFails(self, cls):
    """Tests that DP vectorized optimizers with 'None' unconnected gradients fail.

    Sequence models that have unconnected gradients (with
    'tf.UnconnectedGradients.NONE' passed to tf.GradientTape.jacobian) will
    return a 'None' in the corresponding entry in the Jacobian. To mitigate this
    the 'unconnected_gradients_to_zero' flag is added to the differentially
    private optimizers to support setting these gradients to zero.

    These tests test the various combinations of this flag and the model.

    Args:
      cls: The DP optimizer class to test.
    """

    embedding_model = keras_embedding_model_fn(
        cls,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_seq_output=True,
        unconnected_gradients_to_zero=False)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    self.assertRaisesRegex(
        ValueError,
        'None values not supported',
        embedding_model.fit,
        x=train_data_input_fn(),
        epochs=1,
        verbose=0)

  # Parameters for testing: optimizer.
  @parameterized.named_parameters(
      ('DPSGDVectorized_PooledOutput_UnconnectedGradients',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),)
  def testPooledOutputUnconnectedGradientsAsNonePasses(self, cls):
    """Tests that DP vectorized optimizers with 'None' unconnected gradients fail."""

    embedding_model = keras_embedding_model_fn(
        cls,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_seq_output=False,
        unconnected_gradients_to_zero=False)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    try:
      embedding_model.fit(x=train_data_input_fn(), epochs=1, verbose=0)
    except ValueError:
      # For a 'ValueError' exception the test should record a failure. All
      # other exceptions are errors.
      self.fail('ValueError raised by model.fit().')

  # Parameters for testing: optimizer, use sequence output flag.
  @parameterized.named_parameters(
      ('DPSGDVectorized_SeqOutput_UnconnectedGradientsAreZero',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, True),
      ('DPSGDVectorized_PooledOutput_UnconnectedGradientsAreZero',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, False),
  )
  def testUnconnectedGradientsAsZeroPasses(self, cls, use_seq_output):
    """Tests that DP vectorized optimizers with 'Zero' unconnected gradients pass."""

    embedding_model = keras_embedding_model_fn(
        cls,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_seq_output=use_seq_output,
        unconnected_gradients_to_zero=True)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    try:
      embedding_model.fit(x=train_data_input_fn(), epochs=1, verbose=0)
    except ValueError:
      # For a 'ValueError' exception the test should record a failure. All
      # other exceptions are errors.
      self.fail('ValueError raised by model.fit().')

if __name__ == '__main__':
  tf.test.main()

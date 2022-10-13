# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test the correctness and sparseness of clip_and_aggregate_gradients."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import clip_and_aggregate_gradients as cag


class ClipAndAggregateGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests clip_and_aggreate_gradients."""

  def _get_loss_and_vars_fn(self, n, keepdims=False):
    """Returns the function for creating the loss and variables."""
    # The "model" here consists of both sparse and dense parameters to make sure
    # `clip_and_aggregate_gradients` computes the gradients in the correct way
    # and in the right format. The sparse layer is the embedding layer `emb0`,
    # from which multiple embeddings are gathered, with indices stored
    # in `ind0`. And the dense parameters is the variable var1 which is directly
    # used. The loss is the quadratic loss between the model output and the
    # data stored in `data0` and `data1`. We also add a dummy variable
    # `dummy_var` which does not participate in the loss computation to test
    # the `unconnected` argument.
    emb0 = tf.keras.layers.Embedding(
        4,
        2,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])))
    ind0 = tf.constant([1, 1, 2, 3, 2])
    data0 = tf.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0],
                         [-2.0, -1.0], [-3.0, -2.0]])

    var1 = tf.Variable([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    data1 = tf.constant([[-1.0], [-2.0], [-2.0], [-3.0], [-3.0], [-4.0]])

    dummy_var = tf.Variable(np.array([[1.0]]).astype(np.float64))

    def _loss(val0, val1):
      return 0.5 * tf.reduce_sum(
          input_tensor=tf.math.squared_difference(val0, val1), axis=1)

    def _loss_and_vars_fn():
      # We concatenate the embeddings with some constant values to make sure
      # backprop does only go through those gathered indices.
      val0 = tf.concat([emb0(ind0), tf.constant([[0.0, 0.0]])], axis=0)
      loss = tf.reduce_sum(
          tf.reshape(_loss(data0, val0) + _loss(data1, var1), [n, -1]),
          keepdims=keepdims,
          axis=1)
      return loss, (emb0.embeddings, var1, dummy_var)

    return _loss_and_vars_fn

  def _get_true_grads(self,
                      n,
                      normalize=False,
                      l2_norm_clip=None,
                      agg_method='mean',
                      unconnected='none'):
    # The per-example gradients (or jacobians) below are computed manually.
    # With the (half) quadratic loss, it is the difference between the
    # variable value and the data value.
    grad0 = np.array([[[0., 0.], [-2., -3.], [0., 0.], [0., 0.]],
                      [[0., 0.], [-4., -5.], [0., 0.], [0., 0.]],
                      [[0., 0.], [0., 0.], [-5., -6.], [0., 0.]],
                      [[0., 0.], [0., 0.], [0., 0.], [4., 3.]],
                      [[0., 0.], [0., 0.], [4., 3.], [0., 0.]],
                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
                     dtype=np.float32)
    grad1 = np.array([[[2.], [0.], [0.], [0.], [0.], [0.]],
                      [[0.], [3.], [0.], [0.], [0.], [0.]],
                      [[0.], [0.], [4.], [0.], [0.], [0.]],
                      [[0.], [0.], [0.], [5.], [0.], [0.]],
                      [[0.], [0.], [0.], [0.], [6.], [0.]],
                      [[0.], [0.], [0.], [0.], [0.], [7.]]],
                     dtype=np.float32)
    grad2 = np.array([[[0.]], [[0.]], [[0.]], [[0.]], [[0.]], [[0.]]],
                     dtype=np.float64)

    grads = [
        np.sum(np.reshape(g, (n, -1, g.shape[1], g.shape[2])), axis=1)
        for g in [grad0, grad1, grad2]
    ]

    if normalize or l2_norm_clip is not None:
      if l2_norm_clip is None:
        l2_norm_clip = 1.0
      global_norm = np.sqrt(
          np.sum([
              np.sum(np.square(np.reshape(g, (n, -1))), axis=1) for g in grads
          ],
                 axis=0))
      clip_ratio = l2_norm_clip / np.maximum(global_norm, 1e-8)
      if not normalize:
        clip_ratio = np.minimum(1.0, clip_ratio)
      r = np.reshape(clip_ratio, [n, 1, 1])
      grads = [g * r for g in grads]

    if agg_method == 'sum':
      grads = [np.sum(g, axis=0) for g in grads]
    else:
      grads = [np.mean(g, axis=0) for g in grads]

    if unconnected == 'none':
      grads[2] = None
    return grads

  def _to_dense_array(self, g):
    if g is None:
      return None
    return np.array(tf.convert_to_tensor(g))

  @parameterized.parameters(
      (6, False, None, 'mean', -1, 'none'),
      (6, True, None, 'sum', 1, 'none'),
      (2, False, None, 'sum', 3, 'none'),
      (2, True, 100.0, 'mean', 1, 'zero'),
      (3, False, 1.0, 'sum', 2, 'zero'),
      (1, True, 0.5, 'mean', 3, 'none'),
  )
  def testCorrect(self, n, normalize, l2_norm_clip, agg_method,
                  keep_sparse_threshold, unconnected):
    """Tests the correctness of the computation."""
    loss_and_vars_fn = self._get_loss_and_vars_fn(n)
    true_grads = self._get_true_grads(n, normalize, l2_norm_clip, agg_method,
                                      unconnected)

    with tf.GradientTape() as tape:
      loss, test_vars = loss_and_vars_fn()
      results = cag.clip_and_aggregate_gradients(
          tape,
          loss,
          test_vars,
          normalize=normalize,
          l2_norm_clip=l2_norm_clip,
          aggregate_method=agg_method,
          unconnected_gradients=unconnected,
          keep_sparse_threshold=keep_sparse_threshold)
    for r, t in zip(results, true_grads):
      if t is None:
        self.assertIsNone(r)
      else:
        r = self._to_dense_array(r)
        self.assertAllCloseAccordingToType(r, t)

  @parameterized.parameters(
      (6, True),
      (6, False),
      (1, True),
      (1, False),
  )
  def testTargetShape(self, n, keepdims):
    """Tests target gets vectorized regardless of their original shape."""
    loss_and_vars_fn = self._get_loss_and_vars_fn(n, keepdims)
    true_grads = self._get_true_grads(n)

    with tf.GradientTape() as tape:
      loss, test_vars = loss_and_vars_fn()
      results = cag.clip_and_aggregate_gradients(tape, loss, test_vars)
    for r, t in zip(results, true_grads):
      if t is None:
        self.assertIsNone(r)
      else:
        r = self._to_dense_array(r)
        self.assertAllCloseAccordingToType(r, t)

  @parameterized.parameters(
      (-1),
      (0),
      (4),
      (5),
  )
  def testSparse(self, keep_sparse_threshold):
    """Tests the outcome is in the desired (dense or sparse) tensor form."""
    loss_and_vars_fn = self._get_loss_and_vars_fn(3)
    with tf.GradientTape() as tape:
      loss, test_vars = loss_and_vars_fn()
      results = cag.clip_and_aggregate_gradients(
          tape,
          loss,
          test_vars,
          normalize=False,
          l2_norm_clip=1.0,
          aggregate_method='mean',
          unconnected_gradients='zero',
          keep_sparse_threshold=keep_sparse_threshold)
    grads0, grads1, grads2 = results
    # emb0 has 4 items so grads0 should be in the sparse, i.e.
    # `tf.IndexedSlices`, form iff `keep_sparse_threshold` is in [0, 4].
    if keep_sparse_threshold >= 0 and keep_sparse_threshold <= 4:
      self.assertIsInstance(grads0, tf.IndexedSlices)
      self.assertLen(grads0.indices, 3)
    else:
      self.assertIsInstance(grads0, tf.Tensor)
    # grads1 and grads2 should always be in the dense, i.e. `tf.Tensor`, form.
    self.assertIsInstance(grads1, tf.Tensor)
    self.assertIsInstance(grads2, tf.Tensor)


if __name__ == '__main__':
  tf.test.main()

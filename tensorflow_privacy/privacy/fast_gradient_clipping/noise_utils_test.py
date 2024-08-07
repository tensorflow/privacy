# Copyright 2024, The TensorFlow Authors.
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
from tensorflow_privacy.privacy.fast_gradient_clipping import noise_utils


class NoiseUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      l2_norm_clip=[3.0, 5.0],
      noise_multiplier=[2.0, 4.0],
      batch_size=[1, 2, 10],
      model_fn_reduction=[None, 'auto', 'sum_over_batch_size', 'sum'],
      noise_fn_reduction=[None, 'mean', 'sum'],
  )
  def test_noise_is_computed_correctly(
      self,
      l2_norm_clip,
      noise_multiplier,
      batch_size,
      model_fn_reduction,
      noise_fn_reduction,
  ):
    # Skip invalid combinations.
    if model_fn_reduction is None and noise_fn_reduction is None:
      return
    if model_fn_reduction is not None and noise_fn_reduction is not None:
      return
    # Make an simple model container for storing the loss.
    if model_fn_reduction is not None:
      linear_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
      linear_model.compile(
          loss=tf.keras.losses.MeanSquaredError(reduction=model_fn_reduction)
      )
    else:
      linear_model = None
    # The main computation is done on a deterministic dummy vector.
    num_units = 100
    clipped_grads = [
        tf.expand_dims(np.arange(num_units, dtype=np.float32), axis=-1)
    ]
    noised_grads = noise_utils.add_aggregate_noise(
        clipped_grads,
        batch_size,
        l2_norm_clip,
        noise_multiplier,
        noise_fn_reduction,
        linear_model,
    )
    # The only measure that varies is the standard deviation of the variation.
    scale = (
        1.0
        if noise_fn_reduction == 'sum' or model_fn_reduction == 'sum'
        else 1.0 / batch_size
    )
    computed_std = np.std(noised_grads[0] - clipped_grads[0])
    expected_std = l2_norm_clip * noise_multiplier * scale
    self.assertNear(computed_std, expected_std, 0.1 * expected_std)

  @parameterized.product(
      l2_norm_clip=[3.0, 5.0],
      noise_multiplier=[2.0, 4.0],
      sparse_noise_multiplier=[1.0],
      batch_size=[1, 2, 10],
      model_fn_reduction=[None, 'auto', 'sum_over_batch_size', 'sum'],
      noise_fn_reduction=[None, 'mean', 'sum'],
  )
  def test_sparse_noise_is_computed_correctly(
      self,
      l2_norm_clip,
      noise_multiplier,
      sparse_noise_multiplier,
      batch_size,
      model_fn_reduction,
      noise_fn_reduction,
  ):
    # Skip invalid combinations.
    if model_fn_reduction is None and noise_fn_reduction is None:
      return
    if model_fn_reduction is not None and noise_fn_reduction is not None:
      return
    # Make an simple model container for storing the loss.
    if model_fn_reduction is not None:
      linear_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
      linear_model.compile(
          loss=tf.keras.losses.MeanSquaredError(reduction=model_fn_reduction)
      )
    else:
      linear_model = None
    # The main computation is done on a deterministic dummy vector.
    num_units = 100
    dense_grad = tf.expand_dims(np.arange(num_units, dtype=np.float32), axis=-1)
    sparse_grad = tf.IndexedSlices(
        values=tf.ones((3, 4)),
        indices=tf.constant([0, 3, 5]),
        dense_shape=tf.constant([8, 4]),
    )
    sparse_grad_contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5]],
        values=[10.0, 10.0, 20.0],
        dense_shape=[8],
    )

    sparse_noise_config = noise_utils.SparsityPreservingNoiseConfig(
        sparse_noise_multiplier=sparse_noise_multiplier,
        sparse_selection_threshold=8,
        sparse_contribution_counts=[None, sparse_grad_contribution_counts],
    )

    sparse_noised_grad, dense_noised_grad = noise_utils.add_aggregate_noise(
        clipped_grads=[dense_grad, sparse_grad],
        batch_size=batch_size,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        loss_model=linear_model,
        sparse_noise_config=sparse_noise_config,
    )
    self.assertContainsSubset(
        sparse_grad.indices.numpy().tolist(),
        sparse_noised_grad.indices.numpy().tolist(),
    )
    sparse_noised_grad_dense = tf.scatter_nd(
        tf.reshape(sparse_noised_grad.indices, (-1, 1)),
        sparse_noised_grad.values,
        shape=(8, 4),
    ).numpy()
    sparse_noised_grad_valid_indices = sparse_noised_grad_dense[
        sparse_grad.indices.numpy()
    ]
    sparse_grad_values = sparse_grad.values.numpy()
    self.assertTrue(
        np.all(
            np.not_equal(sparse_noised_grad_valid_indices, sparse_grad_values)
        )
    )
    scale = (
        1.0
        if noise_fn_reduction == 'sum' or model_fn_reduction == 'sum'
        else 1.0 / batch_size
    )
    # The only measure that varies is the standard deviation of the variation.
    expected_std = l2_norm_clip * noise_multiplier * scale

    sparse_computed_std = np.std(
        sparse_noised_grad_valid_indices - sparse_grad_values
    )
    self.assertNear(sparse_computed_std, expected_std, 0.1 * expected_std)

    dense_computed_std = np.std(dense_noised_grad - dense_grad)
    self.assertNear(dense_computed_std, expected_std, 0.1 * expected_std)

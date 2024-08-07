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

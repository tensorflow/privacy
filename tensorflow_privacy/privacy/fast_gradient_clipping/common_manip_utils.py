# Copyright 2023, The TensorFlow Authors.
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
"""A collection of common utility functions for tensor/data manipulation."""

from typing import Optional

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def maybe_add_microbatch_axis(
    x: type_aliases.PackedTensors,
    num_microbatches: Optional[type_aliases.BatchSize],
) -> type_aliases.PackedTensors:
  """Adds the microbatch axis to a collection of tensors.

  Args:
    x: Model output or input tensors.
    num_microbatches: If None, x is returned unchanged. Otherwise, must divide
      the batch size.

  Returns:
    The input tensor x, reshaped from [batch_size, ...] to
    [num_microbatches, batch_size / num_microbatches, ...].
  """
  if num_microbatches is None:
    return x

  def _expand(t):
    with tf.control_dependencies(
        [tf.assert_equal(tf.math.floormod(tf.shape(t)[0], num_microbatches), 0)]
    ):
      return tf.reshape(
          t, tf.concat([[num_microbatches, -1], tf.shape(t)[1:]], axis=0)
      )

  return tf.nest.map_structure(_expand, x)

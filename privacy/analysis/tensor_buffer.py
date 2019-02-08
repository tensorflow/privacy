# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A lightweight fixed-sized buffer for maintaining lists.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TensorBuffer(object):
  """A lightweight fixed-sized buffer for maintaining lists.

  The TensorBuffer accumulates tensors of the given shape into a tensor (whose
  rank is one more than that of the given shape) via calls to `append`. The
  current value of the accumulated tensor can be extracted via the property
  `values`.
  """

  def __init__(self, max_size, shape, dtype=tf.int32, name=None):
    """Initializes the TensorBuffer.

    Args:
      max_size: The maximum size. Attempts to append more than this many rows
        will fail with an exception.
      shape: The shape (as tuple or list) of the tensors to accumulate.
      dtype: The type of the tensors.
      name: A string name for the variable_scope used.

    Raises:
      ValueError: If the shape is empty (specifies scalar shape).
    """
    shape = list(shape)
    self._rank = len(shape)
    if not self._rank:
      raise ValueError('Shape cannot be scalar.')
    shape = [max_size] + shape

    with tf.variable_scope(name):
      self._buffer = tf.Variable(
          initial_value=tf.zeros(shape, dtype),
          trainable=False,
          name='buffer')
      self._size = tf.Variable(
          initial_value=0,
          trainable=False,
          name='size')

  def append(self, value):
    """Appends a new tensor to the end of the buffer.

    Args:
      value: The tensor to append. Must match the shape specified in the
        initializer.

    Returns:
      An op appending the new tensor to the end of the buffer.
    """
    with tf.control_dependencies([
        tf.assert_less(
            self._size,
            tf.shape(self._buffer)[0],
            message='Appending past end of TensorBuffer.'),
        tf.assert_equal(
            tf.shape(value),
            tf.shape(self._buffer)[1:],
            message='Appending value of inconsistent shape.')]):
      with tf.control_dependencies(
          [tf.assign(self._buffer[self._size, :], value)]):
        return tf.assign_add(self._size, 1)

  @property
  def values(self):
    """Returns the accumulated tensor."""
    begin_value = tf.zeros([self._rank + 1], dtype=tf.int32)
    value_size = tf.concat(
        [[self._size], tf.constant(-1, tf.int32, [self._rank])], 0)
    return tf.slice(self._buffer, begin_value, value_size)

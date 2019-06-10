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
"""Unit testing for optimizer.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras import keras_parameterized
from privacy.bolton import model
from privacy.bolton import optimizer as opt
from absl.testing import parameterized
from absl.testing import absltest


class TestOptimizer(OptimizerV2):
  """Optimizer used for testing the Private optimizer"""
  def __init__(self):
    super(TestOptimizer, self).__init__('test')
    self.not_private = 'test'
    self.iterations = tf.Variable(1, dtype=tf.float32)
    self._iterations = tf.Variable(1, dtype=tf.float32)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    return 'test'

  def get_config(self):
    return 'test'

  def from_config(cls, config, custom_objects=None):
    return 'test'

  def _create_slots(self):
    return 'test'

  def _resource_apply_dense(self, grad, handle):
    return 'test'

  def _resource_apply_sparse(self, grad, handle, indices):
    return 'test'

  def get_updates(self, loss, params):
    return 'test'

  def apply_gradients(self, grads_and_vars, name=None):
    return 'test'

  def minimize(self, loss, var_list, grad_loss=None, name=None):
    return 'test'

  def get_gradients(self, loss, params):
    return 'test'

class PrivateTest(keras_parameterized.TestCase):
  """Private Optimizer tests"""
  @parameterized.named_parameters([
      {'testcase_name': 'branch True, beta',
       'fn': 'limit_learning_rate',
       'args': [True,
                tf.Variable(2, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(0.5, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'branch True, gamma',
       'fn': 'limit_learning_rate',
       'args': [True,
                tf.Variable(1, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(1, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'branch False, beta',
       'fn': 'limit_learning_rate',
       'args': [False,
                tf.Variable(2, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(0.5, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'branch False, gamma',
       'fn': 'limit_learning_rate',
       'args': [False,
                tf.Variable(1, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(1, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'getattr',
       'fn': '__getattr__',
       'args': ['dtype'],
       'result': tf.float32,
       'test_attr': None},
  ])
  def test_fn(self, fn, args, result, test_attr):
    private = opt.Private(TestOptimizer())
    res = getattr(private, fn, None)(*args)
    if test_attr is not None:
      res = getattr(private, test_attr, None)
    if hasattr(res, 'numpy') and hasattr(result, 'numpy'):  # both tensors/not
      res = res.numpy()
      result = result.numpy()
    self.assertEqual(res, result)

  @parameterized.named_parameters([
      {'testcase_name': 'fn: get_updates',
       'fn': 'get_updates',
       'args': [0, 0]},
      {'testcase_name': 'fn: get_config',
       'fn': 'get_config',
       'args': []},
      {'testcase_name': 'fn: from_config',
       'fn': 'from_config',
       'args': [0]},
      {'testcase_name': 'fn: _resource_apply_dense',
       'fn': '_resource_apply_dense',
       'args': [1, 1]},
      {'testcase_name': 'fn: _resource_apply_sparse',
       'fn': '_resource_apply_sparse',
       'args': [1, 1, 1]},
      {'testcase_name': 'fn: apply_gradients',
       'fn': 'apply_gradients',
       'args': [1]},
      {'testcase_name': 'fn: minimize',
       'fn': 'minimize',
       'args': [1, 1]},
      {'testcase_name': 'fn: _compute_gradients',
       'fn': '_compute_gradients',
       'args': [1, 1]},
      {'testcase_name': 'fn: get_gradients',
       'fn': 'get_gradients',
       'args': [1, 1]},
  ])
  def test_rerouted_function(self, fn, args):
    optimizer = TestOptimizer()
    optimizer = opt.Private(optimizer)
    self.assertEqual(
        getattr(optimizer, fn, lambda: 'fn not found')(*args),
        'test'
    )

  @parameterized.named_parameters([
      {'testcase_name': 'fn: limit_learning_rate',
       'fn': 'limit_learning_rate',
       'args': [1, 1, 1]}
  ])
  def test_not_reroute_fn(self, fn, args):
    optimizer = TestOptimizer()
    optimizer = opt.Private(optimizer)
    self.assertNotEqual(getattr(optimizer, fn, lambda: 'test')(*args),
                        'test')

  @parameterized.named_parameters([
      {'testcase_name': 'attr: not_private',
       'attr': 'not_private'}
  ])
  def test_reroute_attr(self, attr):
    internal_optimizer = TestOptimizer()
    optimizer = opt.Private(internal_optimizer)
    self.assertEqual(optimizer._internal_optimizer, internal_optimizer)

  @parameterized.named_parameters([
      {'testcase_name': 'attr: _internal_optimizer',
       'attr': '_internal_optimizer'}
  ])
  def test_not_reroute_attr(self, attr):
    internal_optimizer = TestOptimizer()
    optimizer = opt.Private(internal_optimizer)
    self.assertEqual(optimizer._internal_optimizer, internal_optimizer)

if __name__ == '__main__':
  test.main()
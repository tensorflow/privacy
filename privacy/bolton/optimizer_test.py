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
from tensorflow.python.keras.regularizers import L1L2
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Model
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import test_util

from absl.testing import parameterized
from privacy.bolton.loss import StrongConvexMixin
from privacy.bolton import optimizer as opt


class TestModel(Model):
  """
  Bolton episilon-delta model
  Uses 4 key steps to achieve privacy guarantees:
  1. Adds noise to weights after training (output perturbation).
  2. Projects weights to R after each batch
  3. Limits learning rate
  4. Use a strongly convex loss function (see compile)

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et. al.
  """

  def __init__(self, n_classes=2):
    """
    Args:
        n_classes: number of output classes to predict.
        epsilon: level of privacy guarantee
        noise_distribution: distribution to pull weight perturbations from
        weights_initializer: initializer for weights
        seed: random seed to use
        dtype: data type to use for tensors
    """
    super(TestModel, self).__init__(name='bolton', dynamic=False)
    self.n_classes = n_classes
    self.layer_input_shape = (16, 1)
    self.output_layer = tf.keras.layers.Dense(
      self.n_classes,
      input_shape=self.layer_input_shape,
      kernel_regularizer=L1L2(l2=1),
      kernel_initializer='glorot_uniform',
    )


  # def call(self, inputs):
  #   """Forward pass of network
  #
  #   Args:
  #       inputs: inputs to neural network
  #
  #   Returns:
  #
  #   """
  #   return self.output_layer(inputs)


class TestLoss(losses.Loss, StrongConvexMixin):
  """Test loss function for testing Bolton model"""
  def __init__(self, reg_lambda, C, radius_constant, name='test'):
    super(TestLoss, self).__init__(name=name)
    self.reg_lambda = reg_lambda
    self.C = C
    self.radius_constant = radius_constant

  def radius(self):
    """Radius of R-Ball (value to normalize weights to after each batch)

    Returns: radius

    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def gamma(self):
    """ Gamma strongly convex

    Returns: gamma

    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def beta(self, class_weight):
    """Beta smoothess

    Args:
      class_weight: the class weights used.

    Returns: Beta

    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def lipchitz_constant(self, class_weight):
    """ L lipchitz continuous

    Args:
      class_weight: class weights used

    Returns: L

    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def call(self, val0, val1):
    """Loss function that is minimized at the mean of the input points."""
    return 0.5 * tf.reduce_sum(tf.math.squared_difference(val0, val1), axis=1)

  def max_class_weight(self, class_weight):
    if class_weight is None:
      return 1

  def kernel_regularizer(self):
    return L1L2(l2=self.reg_lambda)


class TestOptimizer(OptimizerV2):
  """Optimizer used for testing the Bolton optimizer"""
  def __init__(self):
    super(TestOptimizer, self).__init__('test')
    self.not_private = 'test'
    self.iterations = tf.constant(1, dtype=tf.float32)
    self._iterations = tf.constant(1, dtype=tf.float32)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    return 'test'

  def get_config(self):
    return 'test'

  def from_config(self, config, custom_objects=None):
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

  def limit_learning_rate(self):
    return 'test'

class BoltonOptimizerTest(keras_parameterized.TestCase):
  """Bolton Optimizer tests"""
  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': 'branch beta',
       'fn': 'limit_learning_rate',
       'args': [tf.Variable(2, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(0.5, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'branch gamma',
       'fn': 'limit_learning_rate',
       'args': [tf.Variable(1, dtype=tf.float32),
                tf.Variable(1, dtype=tf.float32)],
       'result': tf.Variable(1, dtype=tf.float32),
       'test_attr': 'learning_rate'},
      {'testcase_name': 'getattr',
       'fn': '__getattr__',
       'args': ['dtype'],
       'result': tf.float32,
       'test_attr': None},
      {'testcase_name': 'project_weights_to_r',
       'fn': 'project_weights_to_r',
       'args': ['dtype'],
       'result': tf.float32,
       'test_attr': None},
  ])
  def test_fn(self, fn, args, result, test_attr):
    """test that a fn of Bolton optimizer is working as expected.

    Args:
      fn: method of Optimizer to test
      args: args to optimizer fn
      result: the expected result
      test_attr: None if the fn returns the test result. Otherwise, this is
                the attribute of Bolton to check against result with.

    """
    tf.random.set_seed(1)
    loss = TestLoss(1, 1, 1)
    private = opt.Bolton(TestOptimizer(), loss)
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
    """ tests that a method of the internal optimizer is correctly routed from
    the Bolton instance to the internal optimizer instance (TestOptimizer,
    here).

    Args:
      fn: fn to test
      args: arguments to that fn
    """
    loss = TestLoss(1, 1, 1)
    optimizer = TestOptimizer()
    optimizer = opt.Bolton(optimizer, loss)
    model = TestModel(2)
    model.compile(optimizer, loss)
    model.layers[0].kernel_initializer(model.layer_input_shape)
    print(model.layers[0].__dict__)
    with optimizer('laplace', 2, model.layers, 1, 1, model.n_classes):
      self.assertEqual(
          getattr(optimizer, fn, lambda: 'fn not found')(*args),
          'test'
      )

  @parameterized.named_parameters([
      {'testcase_name': 'fn: limit_learning_rate',
       'fn': 'limit_learning_rate',
       'args': [1, 1, 1]},
      {'testcase_name': 'fn: project_weights_to_r',
       'fn': 'project_weights_to_r',
       'args': []},
      {'testcase_name': 'fn: get_noise',
       'fn': 'get_noise',
       'args': [1, 1, 1, 1]},
  ])
  def test_not_reroute_fn(self, fn, args):
    """Test that a fn that should not be rerouted to the internal optimizer is
    in face not rerouted.

    Args:
      fn: fn to test
      args: arguments to that fn
    """
    optimizer = TestOptimizer()
    loss = TestLoss(1, 1, 1)
    optimizer = opt.Bolton(optimizer, loss)
    self.assertNotEqual(getattr(optimizer, fn, lambda: 'test')(*args),
                        'test')

  @parameterized.named_parameters([
      {'testcase_name': 'attr: _iterations',
       'attr': '_iterations'}
  ])
  def test_reroute_attr(self, attr):
    """ test that attribute of internal optimizer is correctly rerouted to
    the internal optimizer

    Args:
      attr: attribute to test
      result: result after checking attribute
    """
    loss = TestLoss(1, 1, 1)
    internal_optimizer = TestOptimizer()
    optimizer = opt.Bolton(internal_optimizer, loss)
    self.assertEqual(getattr(optimizer, attr),
                     getattr(internal_optimizer, attr)
                     )

  @parameterized.named_parameters([
    {'testcase_name': 'attr does not exist',
     'attr': '_not_valid'}
  ])
  def test_attribute_error(self, attr):
    """ test that attribute of internal optimizer is correctly rerouted to
    the internal optimizer

    Args:
      attr: attribute to test
      result: result after checking attribute
    """
    loss = TestLoss(1, 1, 1)
    internal_optimizer = TestOptimizer()
    optimizer = opt.Bolton(internal_optimizer, loss)
    with self.assertRaises(AttributeError):
      getattr(optimizer, attr)

if __name__ == '__main__':
  test.main()

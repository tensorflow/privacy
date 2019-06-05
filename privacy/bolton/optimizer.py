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
"""Private Optimizer for bolton method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

_private_attributes = ['_internal_optimizer', 'dtype']


class Private(optimizer_v2.OptimizerV2):
  """
    Private optimizer wraps another tf optimizer to be used
    as the visible optimizer to the tf model. No matter the optimizer
    passed, "Private" enables the bolton model to control the learning rate
    based on the strongly convex loss.
  """
  def __init__(self,
               optimizer: optimizer_v2.OptimizerV2,
               dtype=tf.float32
               ):
    """Constructor.

    Args:
        optimizer: Optimizer_v2 or subclass to be used as the optimizer
                    (wrapped).
    """
    self._internal_optimizer = optimizer
    self.dtype = dtype

  def get_config(self):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.get_config()

  def limit_learning_rate(self, is_eager, beta, gamma):
    """Implements learning rate limitation that is required by the bolton
    method for sensitivity bounding of the strongly convex function.
    Sets the learning rate to the min(1/beta, 1/(gamma*t))

    Args:
        is_eager: Whether the model is running in eager mode
        beta: loss function beta-smoothness
        gamma: loss function gamma-strongly convex

    Returns: None

    """
    numerator = tf.Variable(initial_value=1, dtype=self.dtype)
    t = tf.cast(self._iterations, self.dtype)
    # will exist on the internal optimizer
    pred = numerator / beta < numerator / (gamma * t)
    if is_eager:  # check eagerly
      if pred:
        self.learning_rate = numerator / beta
      else:
        self.learning_rate = numerator / (gamma * t)
    else:
      if pred:
        self.learning_rate = numerator / beta
      else:
        self.learning_rate = numerator / (gamma * t)

  def from_config(self, config, custom_objects=None):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.from_config(
        config,
        custom_objects=custom_objects
    )

  def __getattr__(self, name):
    """return _internal_optimizer off self instance, and everything else
    from the _internal_optimizer instance.

    Args:
        name:

    Returns: attribute from Private if specified to come from self, else
            from _internal_optimizer.

    """
    if name in _private_attributes:
      return getattr(self, name)
    optim = object.__getattribute__(self, '_internal_optimizer')
    return object.__getattribute__(optim, name)

  def __setattr__(self, key, value):
    """ Set attribute to self instance if its the internal optimizer.
    Reroute everything else to the _internal_optimizer.

    Args:
        key: attribute name
        value: attribute value

    Returns:

    """
    if key in _private_attributes:
      object.__setattr__(self, key, value)
    else:
      setattr(self._internal_optimizer, key, value)

  def _resource_apply_dense(self, grad, handle):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer._resource_apply_dense(grad, handle)

  def _resource_apply_sparse(self, grad, handle, indices):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer._resource_apply_sparse(
        grad,
        handle,
        indices
    )

  def get_updates(self, loss, params):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.get_updates(loss, params)

  def apply_gradients(self, grads_and_vars, name: str = None):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.apply_gradients(
        grads_and_vars,
        name=name
    )

  def minimize(self,
               loss,
               var_list,
               grad_loss: bool = None,
               name: str = None
               ):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.minimize(
        loss,
        var_list,
        grad_loss,
        name
    )

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer._compute_gradients(
        loss,
        var_list,
        grad_loss=grad_loss
    )

  def get_gradients(self, loss, params):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.get_gradients(loss, params)

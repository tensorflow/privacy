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
"""Bolton Optimizer for bolton method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from privacy.bolton.loss import StrongConvexMixin

_accepted_distributions = ['laplace']


class Bolton(optimizer_v2.OptimizerV2):
  """
    Bolton optimizer wraps another tf optimizer to be used
    as the visible optimizer to the tf model. No matter the optimizer
    passed, "Bolton" enables the bolton model to control the learning rate
    based on the strongly convex loss.

    For more details on the strong convexity requirements, see:
    Bolt-on Differential Privacy for Scalable Stochastic Gradient
    Descent-based Analytics by Xi Wu et. al.
  """
  def __init__(self,
               optimizer: optimizer_v2.OptimizerV2,
               loss: StrongConvexMixin,
               dtype=tf.float32,
               ):
    """Constructor.

    Args:
        optimizer: Optimizer_v2 or subclass to be used as the optimizer
                    (wrapped).
    """

    if not isinstance(loss, StrongConvexMixin):
      raise ValueError("loss function must be a Strongly Convex and therfore"
                       "extend the StrongConvexMixin.")
    self._private_attributes = ['_internal_optimizer',
                                'dtype',
                                'noise_distribution',
                                'epsilon',
                                'loss',
                                'class_weights',
                                'input_dim',
                                'n_samples',
                                'n_classes',
                                'layers',
                                '_model'
                                ]
    self._internal_optimizer = optimizer
    self.dtype = dtype
    self.loss = loss

  def get_config(self):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.get_config()

  def project_weights_to_r(self, force=False):
    """helper method to normalize the weights to the R-ball.

    Args:
        r: radius of "R-Ball". Scalar to normalize to.
        force: True to normalize regardless of previous weight values.
                False to check if weights > R-ball and only normalize then.

    Returns:

    """
    r = self.loss.radius()
    for layer in self.layers:
      if tf.executing_eagerly():
        weight_norm = tf.norm(layer.kernel, axis=0)
        if force:
          layer.kernel = layer.kernel / (weight_norm / r)
        elif tf.reduce_sum(tf.cast(weight_norm > r, dtype=self.dtype)) > 0:
          layer.kernel = layer.kernel / (weight_norm / r)
      else:
        weight_norm = tf.norm(layer.kernel, axis=0)
        if force:
          layer.kernel = layer.kernel / (weight_norm / r)
        else:
          layer.kernel = tf.cond(
              tf.reduce_sum(tf.cast(weight_norm > r, dtype=self.dtype)) > 0,
              lambda: layer.kernel / (weight_norm / r),
              lambda: layer.kernel
          )

  def get_noise(self, data_size, input_dim, output_dim, class_weight):
    """Sample noise to be added to weights for privacy guarantee

    Args:
        distribution: the distribution type to pull noise from
        data_size: the number of samples

    Returns: noise in shape of layer's weights to be added to the weights.

    """
    loss = self.loss
    distribution = self.noise_distribution.lower()
    if distribution == _accepted_distributions[0]:  # laplace
      per_class_epsilon = self.epsilon / (output_dim)
      l2_sensitivity = (2 *
                        loss.lipchitz_constant(class_weight)) / \
                       (loss.gamma() * data_size)
      unit_vector = tf.random.normal(shape=(input_dim, output_dim),
                                     mean=0,
                                     seed=1,
                                     stddev=1.0,
                                     dtype=self.dtype)
      unit_vector = unit_vector / tf.math.sqrt(
          tf.reduce_sum(tf.math.square(unit_vector), axis=0)
      )

      beta = l2_sensitivity / per_class_epsilon
      alpha = input_dim  # input_dim
      gamma = tf.random.gamma([output_dim],
                              alpha,
                              beta=1 / beta,
                              seed=1,
                              dtype=self.dtype
                              )
      return unit_vector * gamma
    raise NotImplementedError('Noise distribution: {0} is not '
                              'a valid distribution'.format(distribution))

  def limit_learning_rate(self, beta, gamma):
    """Implements learning rate limitation that is required by the bolton
    method for sensitivity bounding of the strongly convex function.
    Sets the learning rate to the min(1/beta, 1/(gamma*t))

    Args:
        is_eager: Whether the model is running in eager mode
        beta: loss function beta-smoothness
        gamma: loss function gamma-strongly convex

    Returns: None

    """
    numerator = tf.constant(1, dtype=self.dtype)
    t = tf.cast(self._iterations, self.dtype)
    # will exist on the internal optimizer
    if numerator / beta < numerator / (gamma * t):
      self.learning_rate = numerator / beta
    else:
      self.learning_rate = numerator / (gamma * t)

  def from_config(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer.from_config(*args, **kwargs)

  def __getattr__(self, name):
    """return _internal_optimizer off self instance, and everything else
    from the _internal_optimizer instance.

    Args:
        name:

    Returns: attribute from Bolton if specified to come from self, else
            from _internal_optimizer.

    """
    if name == '_private_attributes':
      return getattr(self, name)
    elif name in self._private_attributes:
      return getattr(self, name)
    optim = object.__getattribute__(self, '_internal_optimizer')
    try:
      return object.__getattribute__(optim, name)
    except AttributeError:
      raise AttributeError("Neither '{0}' nor '{1}' object has attribute '{2}'"
                           "".format(
          self.__class__.__name__,
          self._internal_optimizer.__class__.__name__,
          name
                                     )
                           )

  def __setattr__(self, key, value):
    """ Set attribute to self instance if its the internal optimizer.
    Reroute everything else to the _internal_optimizer.

    Args:
        key: attribute name
        value: attribute value

    Returns:

    """
    if key == '_private_attributes':
      object.__setattr__(self, key, value)
    elif key in key in self._private_attributes:
      object.__setattr__(self, key, value)
    else:
      setattr(self._internal_optimizer, key, value)

  def _resource_apply_dense(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer._resource_apply_dense(*args, **kwargs)

  def _resource_apply_sparse(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    return self._internal_optimizer._resource_apply_sparse(*args, **kwargs)

  def get_updates(self, loss, params):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    # self.layers = params
    out = self._internal_optimizer.get_updates(loss, params)
    self.limit_learning_rate(self.loss.beta(self.class_weights),
                             self.loss.gamma()
                             )
    self.project_weights_to_r()
    return out

  def apply_gradients(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    # grads_and_vars = kwargs.get('grads_and_vars', None)
    # grads_and_vars = optimizer_v2._filter_grads(grads_and_vars)
    # var_list = [v for (_, v) in grads_and_vars]
    # self.layers = var_list
    out = self._internal_optimizer.apply_gradients(*args, **kwargs)
    self.limit_learning_rate(self.loss.beta(self.class_weights),
                             self.loss.gamma()
                             )
    self.project_weights_to_r()
    return out

  def minimize(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    # self.layers = kwargs.get('var_list', None)
    out = self._internal_optimizer.minimize(*args, **kwargs)
    self.limit_learning_rate(self.loss.beta(self.class_weights),
                             self.loss.gamma()
                             )
    self.project_weights_to_r()
    return out

  def _compute_gradients(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    # self.layers = kwargs.get('var_list', None)
    return self._internal_optimizer._compute_gradients(*args, **kwargs)

  def get_gradients(self, *args, **kwargs):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer.
    """
    # self.layers = kwargs.get('params', None)
    return self._internal_optimizer.get_gradients(*args, **kwargs)

  def __enter__(self):
    noise_distribution = self.noise_distribution
    epsilon = self.epsilon
    class_weights = self.class_weights
    n_samples = self.n_samples
    if noise_distribution not in _accepted_distributions:
      raise ValueError('Detected noise distribution: {0} not one of: {1} valid'
                       'distributions'.format(noise_distribution,
                                              _accepted_distributions))
    self.noise_distribution = noise_distribution
    self.epsilon = epsilon
    self.class_weights = class_weights
    self.n_samples = n_samples
    return self

  def __call__(self,
               noise_distribution,
               epsilon,
               layers,
               class_weights,
               n_samples,
               n_classes,
               ):
    """

    Args:
      noise_distribution: the noise distribution to pick.
                          see _accepted_distributions and get_noise for
                          possible values.
      epsilon: privacy parameter. Lower gives more privacy but less utility.
      class_weights: class_weights used
      n_samples number of rows/individual samples in the training set
      n_classes: number of output classes
      layers: list of Keras/Tensorflow layers.
    """
    if epsilon <= 0:
      raise ValueError('Detected epsilon: {0}. '
                       'Valid range is 0 < epsilon <inf'.format(epsilon))
    self.noise_distribution = noise_distribution
    self.epsilon = epsilon
    self.class_weights = class_weights
    self.n_samples = n_samples
    self.n_classes = n_classes
    self.layers = layers
    return self

  def __exit__(self, *args):
    """Exit call from with statement.
        used to

        1.reset the model and fit parameters passed to the optimizer
          to enable the Bolton Privacy guarantees. These are reset to ensure
          that any future calls to fit with the same instance of the optimizer
          will properly error out.

        2.call post-fit methods normalizing/projecting the model weights and
          adding noise to the weights.


    """
    # for param in self.layers:
    #   if param.name.find('kernel') != -1 or param.name.find('weight') != -1:
    #     input_dim = param.numpy().shape[0]
    #     print(param)
    #     noise = -1 * self.get_noise(self.n_samples,
    #                                 input_dim,
    #                                 self.n_classes,
    #                                 self.class_weights
    #                                 )
    #     print(tf.math.subtract(param, noise))
    #     param.assign(tf.math.subtract(param, noise))
    self.project_weights_to_r(True)
    for layer in self.layers:
      input_dim, output_dim = layer.kernel.shape
      noise = self.get_noise(self.n_samples,
                             input_dim,
                             output_dim,
                             self.class_weights
                             )
      layer.kernel = tf.math.add(layer.kernel, noise)
    self.noise_distribution = None
    self.epsilon = -1
    self.class_weights = None
    self.n_samples = None
    self.input_dim = None
    self.n_classes = None
    self.layers = None

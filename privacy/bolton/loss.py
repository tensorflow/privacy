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
"""Loss functions for bolton method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.framework import ops as _ops


class StrongConvexLoss(losses.Loss):
  """
  Strong Convex Loss base class for any loss function that will be used with
  Bolton model. Subclasses must be strongly convex and implement the
  associated constants. They must also conform to the requirements of tf losses
  (see super class)
  """
  def __init__(self,
               reg_lambda: float,
               c: float,
               radius_constant: float = 1,
               reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name: str = None,
               dtype=tf.float32,
               **kwargs):
    """
    Args:
      reg_lambda: Weight regularization constant
      c: Additional constant for strongly convex convergence. Acts
          as a global weight.
      radius_constant: constant defining the length of the radius
      reduction: reduction type to use. See super class
      name: Name of the loss instance
      dtype: tf datatype to use for tensor conversions.
    """
    super(StrongConvexLoss, self).__init__(reduction=reduction,
                                           name=name,
                                           **kwargs)
    self._sample_weight = tf.Variable(initial_value=c,
                                      trainable=False,
                                      dtype=tf.float32)
    self._reg_lambda = reg_lambda
    self.radius_constant = tf.Variable(initial_value=radius_constant,
                                       trainable=False,
                                       dtype=tf.float32)
    self.dtype = dtype

  def radius(self):
    """Radius of R-Ball (value to normalize weights to after each batch)

    Returns: radius

    """
    raise NotImplementedError("Radius not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def gamma(self):
    """ Gamma strongly convex

    Returns: gamma

    """
    raise NotImplementedError("Gamma not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def beta(self, class_weight):
    """Beta smoothess

    Args:
      class_weight: the class weights used.

    Returns: Beta

    """
    raise NotImplementedError("Beta not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def lipchitz_constant(self, class_weight):
    """ L lipchitz continuous

    Args:
      class_weight: class weights used

    Returns: L

    """
    raise NotImplementedError("lipchitz constant not implemented for "
                              "StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def reg_lambda(self, convert_to_tensor: bool = False):
    """ returns the lambda weight regularization constant, as a tensor if
    desired

    Args:
      convert_to_tensor: True to convert to tensor, False to leave as
                            python numeric.

    Returns: reg_lambda

    """
    if convert_to_tensor:
      return _ops.convert_to_tensor_v2(self._reg_lambda, dtype=self.dtype)
    return self._reg_lambda

  def max_class_weight(self, class_weight):
    class_weight = _ops.convert_to_tensor_v2(class_weight, dtype=self.dtype)
    return tf.math.reduce_max(class_weight)


class Huber(StrongConvexLoss, losses.Huber):
  """Strong Convex version of huber loss using l2 weight regularization.
  """
  def __init__(self,
               reg_lambda: float,
               c: float,
               radius_constant: float,
               delta: float,
               reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name: str = 'huber',
               dtype=tf.float32):
    """Constructor. Passes arguments to StrongConvexLoss and Huber Loss.

    Args:
      reg_lambda: Weight regularization constant
      c: Additional constant for strongly convex convergence. Acts
          as a global weight.
      radius_constant: constant defining the length of the radius
      delta: delta value in huber loss.  When to switch from quadratic to
            absolute deviation.
      reduction: reduction type to use. See super class
      name: Name of the loss instance
      dtype: tf datatype to use for tensor conversions.

    Returns:
      Loss values per sample.
    """
    # self.delta = tf.Variable(initial_value=delta, trainable=False)
    super(Huber, self).__init__(
        reg_lambda,
        c,
        radius_constant,
        delta=delta,
        name=name,
        reduction=reduction,
        dtype=dtype
    )

  def call(self, y_true, y_pred):
    """Compute loss

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    return super(Huber, self).call(y_true, y_pred, **self._fn_kwargs) * \
           self._sample_weight

  def radius(self):
    """See super class.
    """
    return self.radius_constant / self.reg_lambda(True)

  def gamma(self):
    """See super class.
    """
    return self.reg_lambda(True)

  def beta(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight)
    return self._sample_weight * max_class_weight / \
           (self.delta * tf.Variable(initial_value=2, trainable=False)) + \
           self.reg_lambda(True)

  def lipchitz_constant(self, class_weight):
    """See super class.
    """
    # if class_weight is provided,
    # it should be a vector of the same size of number of classes
    max_class_weight = self.max_class_weight(class_weight)
    lc = self._sample_weight * max_class_weight + \
         self.reg_lambda(True) * self.radius()
    return lc


class BinaryCrossentropy(StrongConvexLoss, losses.BinaryCrossentropy):
  """
  Strong Convex version of BinaryCrossentropy loss using l2 weight
  regularization.
  """
  def __init__(self,
               reg_lambda: float,
               c: float,
               radius_constant: float,
               from_logits: bool = True,
               label_smoothing: float = 0,
               reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name: str = 'binarycrossentropy',
               dtype=tf.float32):
    """
    Args:
      reg_lambda: Weight regularization constant
      c: Additional constant for strongly convex convergence. Acts
          as a global weight.
      radius_constant: constant defining the length of the radius
      reduction: reduction type to use. See super class
      label_smoothing: amount of smoothing to perform on labels
                      relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x)
      name: Name of the loss instance
      dtype: tf datatype to use for tensor conversions.
    """
    super(BinaryCrossentropy, self).__init__(reg_lambda,
                                             c,
                                             radius_constant,
                                             reduction=reduction,
                                             name=name,
                                             from_logits=from_logits,
                                             label_smoothing=label_smoothing,
                                             dtype=dtype
                                             )
    self.radius_constant = radius_constant

  def call(self, y_true, y_pred):
    """Compute loss

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
      """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred
    )
    loss = loss * self._sample_weight
    return loss

  def radius(self):
    """See super class.
    """
    return self.radius_constant / self.reg_lambda(True)

  def gamma(self):
    """See super class.
    """
    return self.reg_lambda(True)

  def beta(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight)
    return self._sample_weight * max_class_weight + self.reg_lambda(True)

  def lipchitz_constant(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight)
    return self._sample_weight * max_class_weight + \
           self.reg_lambda(True) * self.radius()

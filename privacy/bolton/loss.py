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
from tensorflow.python.keras.regularizers import L1L2


class StrongConvexMixin:
  """
  Strong Convex Mixin base class for any loss function that will be used with
  Bolton model. Subclasses must be strongly convex and implement the
  associated constants. They must also conform to the requirements of tf losses
  (see super class).

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et. al.
  """

  def radius(self):
    """Radius, R, of the hypothesis space W.
    W is a convex set that forms the hypothesis space.

    Returns: R

    """
    raise NotImplementedError("Radius not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def gamma(self):
    """ Strongly convexity, gamma

    Returns: gamma

    """
    raise NotImplementedError("Gamma not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def beta(self, class_weight):
    """Smoothness, beta

    Args:
      class_weight: the class weights as scalar or 1d tensor, where its
                    dimensionality is equal to the number of outputs.

    Returns: Beta

    """
    raise NotImplementedError("Beta not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def lipchitz_constant(self, class_weight):
    """Lipchitz constant, L

    Args:
      class_weight: class weights used

    Returns: L

    """
    raise NotImplementedError("lipchitz constant not implemented for "
                              "StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def kernel_regularizer(self):
    """returns the kernel_regularizer to be used. Any subclass should override
      this method if they want a kernel_regularizer (if required for
      the loss function to be StronglyConvex

    :return: None or kernel_regularizer layer
    """
    return None

  def max_class_weight(self, class_weight, dtype):
    """the maximum weighting in class weights (max value) as a scalar tensor

    Args:
      class_weight: class weights used
      dtype: the data type for tensor conversions.

    Returns: maximum class weighting as tensor scalar

    """
    class_weight = _ops.convert_to_tensor_v2(class_weight, dtype)
    return tf.math.reduce_max(class_weight)


class StrongConvexHuber(losses.Loss, StrongConvexMixin):
  """Strong Convex version of Huber loss using l2 weight regularization.
  """

  def __init__(self,
               reg_lambda: float,
               C: float,
               radius_constant: float,
               delta: float,
               reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """Constructor.

    Args:
      reg_lambda: Weight regularization constant
      C: Penalty parameter C of the loss term
      radius_constant: constant defining the length of the radius
      delta: delta value in huber loss.  When to switch from quadratic to
            absolute deviation.
      reduction: reduction type to use. See super class
      name: Name of the loss instance
      dtype: tf datatype to use for tensor conversions.

    Returns:
      Loss values per sample.
    """
    if C <= 0:
      raise ValueError('c: {0}, should be >= 0'.format(C))
    if reg_lambda <= 0:
      raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
    if radius_constant <= 0:
      raise ValueError('radius_constant: {0}, should be >= 0'.format(
          radius_constant
      ))
    if delta <= 0:
      raise ValueError('delta: {0}, should be >= 0'.format(
          delta
      ))
    self.C = C  # pylint: disable=invalid-name
    self.delta = delta
    self.radius_constant = radius_constant
    self.dtype = dtype
    self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
    super(StrongConvexHuber, self).__init__(
        name='huber',
        reduction=reduction,
    )

  def call(self, y_true, y_pred):
    """Compute loss

    Args:
      y_true: Ground truth values. One hot encoded using -1 and 1.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    # return super(StrongConvexHuber, self).call(y_true, y_pred) * self._sample_weight
    h = self.delta
    z = y_pred * y_true
    one = tf.constant(1, dtype=self.dtype)
    four = tf.constant(4, dtype=self.dtype)

    if z > one + h:
      return _ops.convert_to_tensor_v2(0, dtype=self.dtype)
    elif tf.math.abs(one - z) <= h:
      return one / (four * h) * tf.math.pow(one + h - z, 2)
    elif z < one - h:
      return one - z
    raise ValueError('')  # shouldn't be possible to get here.

  def radius(self):
    """See super class.
    """
    return self.radius_constant / self.reg_lambda

  def gamma(self):
    """See super class.
    """
    return self.reg_lambda

  def beta(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    delta = _ops.convert_to_tensor_v2(self.delta,
                                      dtype=self.dtype
                                      )
    return self.C * max_class_weight / (delta *
                                        tf.constant(2, dtype=self.dtype)) + \
           self.reg_lambda

  def lipchitz_constant(self, class_weight):
    """See super class.
    """
    # if class_weight is provided,
    # it should be a vector of the same size of number of classes
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    lc = self.C * max_class_weight + \
         self.reg_lambda * self.radius()
    return lc

  def kernel_regularizer(self):
    """
      l2 loss using reg_lambda as the l2 term (as desired). Required for
      this loss function to be strongly convex.
    :return:
    """
    return L1L2(l2=self.reg_lambda/2)


class StrongConvexBinaryCrossentropy(
    losses.BinaryCrossentropy,
    StrongConvexMixin
):
  """
  Strong Convex version of BinaryCrossentropy loss using l2 weight
  regularization.
  """

  def __init__(self,
               reg_lambda: float,
               C: float,
               radius_constant: float,
               from_logits: bool = True,
               label_smoothing: float = 0,
               reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """
    Args:
      reg_lambda: Weight regularization constant
      C: Penalty parameter C of the loss term
      radius_constant: constant defining the length of the radius
      reduction: reduction type to use. See super class
      label_smoothing: amount of smoothing to perform on labels
                      relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x).
                      Note, the impact of this parameter's effect on privacy
                      is not known and thus the default should be used.
      name: Name of the loss instance
      dtype: tf datatype to use for tensor conversions.
    """
    if reg_lambda <= 0:
      raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
    if C <= 0:
      raise ValueError('c: {0}, should be >= 0'.format(C))
    if radius_constant <= 0:
      raise ValueError('radius_constant: {0}, should be >= 0'.format(
          radius_constant
      ))
    self.dtype = dtype
    self.C = C  # pylint: disable=invalid-name
    self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
    super(StrongConvexBinaryCrossentropy, self).__init__(
        reduction=reduction,
        name='binarycrossentropy',
        from_logits=from_logits,
        label_smoothing=label_smoothing,
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
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #   labels=y_true,
    #   logits=y_pred
    # )
    loss = super(StrongConvexBinaryCrossentropy, self).call(y_true, y_pred)
    loss = loss * self.C
    return loss

  def radius(self):
    """See super class.
    """
    return self.radius_constant / self.reg_lambda

  def gamma(self):
    """See super class.
    """
    return self.reg_lambda

  def beta(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    return self.C * max_class_weight + self.reg_lambda

  def lipchitz_constant(self, class_weight):
    """See super class.
    """
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    return self.C * max_class_weight + self.reg_lambda * self.radius()

  def kernel_regularizer(self):
    """
      l2 loss using reg_lambda as the l2 term (as desired). Required for
      this loss function to be strongly convex.
    :return:
    """
    return L1L2(l2=self.reg_lambda/2)


# class StrongConvexSparseCategoricalCrossentropy(
#     losses.CategoricalCrossentropy,
#     StrongConvexMixin
# ):
#   """
#   Strong Convex version of CategoricalCrossentropy loss using l2 weight
#   regularization.
#   """
#
#   def __init__(self,
#                reg_lambda: float,
#                C: float,
#                radius_constant: float,
#                from_logits: bool = True,
#                label_smoothing: float = 0,
#                reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#                name: str = 'binarycrossentropy',
#                dtype=tf.float32):
#     """
#     Args:
#       reg_lambda: Weight regularization constant
#       C: Penalty parameter C of the loss term
#       radius_constant: constant defining the length of the radius
#       reduction: reduction type to use. See super class
#       label_smoothing: amount of smoothing to perform on labels
#                       relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x)
#       name: Name of the loss instance
#       dtype: tf datatype to use for tensor conversions.
#     """
#     if reg_lambda <= 0:
#       raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
#     if C <= 0:
#       raise ValueError('c: {0}, should be >= 0'.format(C))
#     if radius_constant <= 0:
#       raise ValueError('radius_constant: {0}, should be >= 0'.format(
#         radius_constant
#       ))
#
#     self.C = C
#     self.dtype = dtype
#     self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
#     super(StrongConvexSparseCategoricalCrossentropy, self).__init__(
#         reduction=reduction,
#         name=name,
#         from_logits=from_logits,
#         label_smoothing=label_smoothing,
#     )
#     self.radius_constant = radius_constant
#
#   def call(self, y_true, y_pred):
#     """Compute loss
#
#         Args:
#           y_true: Ground truth values.
#           y_pred: The predicted values.
#
#         Returns:
#           Loss values per sample.
#       """
#     loss = super()
#     loss = loss * self.C
#     return loss
#
#   def radius(self):
#     """See super class.
#     """
#     return self.radius_constant / self.reg_lambda
#
#   def gamma(self):
#     """See super class.
#     """
#     return self.reg_lambda
#
#   def beta(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda
#
#   def lipchitz_constant(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda * self.radius()
#
#   def kernel_regularizer(self):
#     """
#       l2 loss using reg_lambda as the l2 term (as desired). Required for
#       this loss function to be strongly convex.
#     :return:
#     """
#     return L1L2(l2=self.reg_lambda)
#
# class StrongConvexSparseCategoricalCrossentropy(
#     losses.SparseCategoricalCrossentropy,
#     StrongConvexMixin
# ):
#   """
#   Strong Convex version of SparseCategoricalCrossentropy loss using l2 weight
#   regularization.
#   """
#
#   def __init__(self,
#                reg_lambda: float,
#                C: float,
#                radius_constant: float,
#                from_logits: bool = True,
#                label_smoothing: float = 0,
#                reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#                name: str = 'binarycrossentropy',
#                dtype=tf.float32):
#     """
#     Args:
#       reg_lambda: Weight regularization constant
#       C: Penalty parameter C of the loss term
#       radius_constant: constant defining the length of the radius
#       reduction: reduction type to use. See super class
#       label_smoothing: amount of smoothing to perform on labels
#                       relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x)
#       name: Name of the loss instance
#       dtype: tf datatype to use for tensor conversions.
#     """
#     if reg_lambda <= 0:
#       raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
#     if C <= 0:
#       raise ValueError('c: {0}, should be >= 0'.format(C))
#     if radius_constant <= 0:
#       raise ValueError('radius_constant: {0}, should be >= 0'.format(
#         radius_constant
#       ))
#
#     self.C = C
#     self.dtype = dtype
#     self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
#     super(StrongConvexHuber, self).__init__(reduction=reduction,
#                                              name=name,
#                                              from_logits=from_logits,
#                                              label_smoothing=label_smoothing,
#                                              )
#     self.radius_constant = radius_constant
#
#   def call(self, y_true, y_pred):
#     """Compute loss
#
#         Args:
#           y_true: Ground truth values.
#           y_pred: The predicted values.
#
#         Returns:
#           Loss values per sample.
#       """
#     loss = super()
#     loss = loss * self.C
#     return loss
#
#   def radius(self):
#     """See super class.
#     """
#     return self.radius_constant / self.reg_lambda
#
#   def gamma(self):
#     """See super class.
#     """
#     return self.reg_lambda
#
#   def beta(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda
#
#   def lipchitz_constant(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda * self.radius()
#
#   def kernel_regularizer(self):
#     """
#       l2 loss using reg_lambda as the l2 term (as desired). Required for
#       this loss function to be strongly convex.
#     :return:
#     """
#     return L1L2(l2=self.reg_lambda)
#
#
# class StrongConvexCategoricalCrossentropy(
#     losses.CategoricalCrossentropy,
#     StrongConvexMixin
# ):
#   """
#   Strong Convex version of CategoricalCrossentropy loss using l2 weight
#   regularization.
#   """
#
#   def __init__(self,
#                reg_lambda: float,
#                C: float,
#                radius_constant: float,
#                from_logits: bool = True,
#                label_smoothing: float = 0,
#                reduction: str = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#                name: str = 'binarycrossentropy',
#                dtype=tf.float32):
#     """
#     Args:
#       reg_lambda: Weight regularization constant
#       C: Penalty parameter C of the loss term
#       radius_constant: constant defining the length of the radius
#       reduction: reduction type to use. See super class
#       label_smoothing: amount of smoothing to perform on labels
#                       relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x)
#       name: Name of the loss instance
#       dtype: tf datatype to use for tensor conversions.
#     """
#     if reg_lambda <= 0:
#       raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
#     if C <= 0:
#       raise ValueError('c: {0}, should be >= 0'.format(C))
#     if radius_constant <= 0:
#       raise ValueError('radius_constant: {0}, should be >= 0'.format(
#         radius_constant
#       ))
#
#     self.C = C
#     self.dtype = dtype
#     self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
#     super(StrongConvexHuber, self).__init__(reduction=reduction,
#                                              name=name,
#                                              from_logits=from_logits,
#                                              label_smoothing=label_smoothing,
#                                              )
#     self.radius_constant = radius_constant
#
#   def call(self, y_true, y_pred):
#     """Compute loss
#
#         Args:
#           y_true: Ground truth values.
#           y_pred: The predicted values.
#
#         Returns:
#           Loss values per sample.
#       """
#     loss = super()
#     loss = loss * self.C
#     return loss
#
#   def radius(self):
#     """See super class.
#     """
#     return self.radius_constant / self.reg_lambda
#
#   def gamma(self):
#     """See super class.
#     """
#     return self.reg_lambda
#
#   def beta(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda
#
#   def lipchitz_constant(self, class_weight):
#     """See super class.
#     """
#     max_class_weight = self.max_class_weight(class_weight, self.dtype)
#     return self.C * max_class_weight + self.reg_lambda * self.radius()
#
#   def kernel_regularizer(self):
#     """
#       l2 loss using reg_lambda as the l2 term (as desired). Required for
#       this loss function to be strongly convex.
#     :return:
#     """
#     return L1L2(l2=self.reg_lambda)

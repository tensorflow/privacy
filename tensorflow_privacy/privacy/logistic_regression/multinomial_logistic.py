# Copyright 2021, The TensorFlow Authors.
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
"""Implementation of differentially private multinomial logistic regression.

Algorithms include:

- Based on the differentially private objective perturbation method of Kifer et
al. (Colt 2012): http://proceedings.mlr.press/v23/kifer12/kifer12.pdf
Their algorithm can be used for convex optimization problems in general, and in
the case of multinomial logistic regression in particular.

- Training procedure based on the Differentially Private Stochastic Gradient
Descent (DP-SGD) implementation in TensorFlow Privacy, which is itself based on
the algorithm of Abadi et al.: https://arxiv.org/pdf/1607.00133.pdf%20.
"""

import math
from typing import List, Optional, Tuple

import dp_accounting
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.logistic_regression import datasets
from tensorflow_privacy.privacy.logistic_regression import single_layer_softmax
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras


@tf.keras.utils.register_keras_serializable(package='Custom', name='Kifer')
class KiferRegularizer(tf.keras.regularizers.Regularizer):
  """Class corresponding to the regularizer in Algorithm 1 of Kifer et al.

  Attributes:
    l2_regularizer: scalar coefficient for l2-regularization term.
    num_train: number of training examples.
    b: tensor of shape (d,num_classes) linearly translating the objective.
  """

  def __init__(self, num_train: int, dimension: int, epsilon: float,
               delta: float, num_classes: int, input_clipping_norm: float):
    self._num_train = num_train
    (self._l2_regularizer,
     variance) = self.logistic_objective_perturbation_parameters(
         num_train, epsilon, delta, num_classes, input_clipping_norm)
    self._b = tf.random.normal(
        shape=[dimension, num_classes],
        mean=0.0,
        stddev=math.sqrt(variance),
        dtype=tf.dtypes.float32)

  def __call__(self, x):
    return (tf.reduce_sum(self._l2_regularizer * tf.square(x)) +
            (1 / self._num_train) * tf.reduce_sum(tf.multiply(x, self._b)))

  def get_config(self):
    return {
        'l2_regularizer': self._l2_regularizer,
        'num_train': self._num_train,
        'b': self._b
    }

  def logistic_objective_perturbation_parameters(
      self, num_train: int, epsilon: float, delta: float, num_classes: int,
      input_clipping_norm: float) -> Tuple[float, float]:
    """Computes l2-regularization coefficient and Gaussian noise variance.

      The setting is based on Algorithm 1 of Kifer et al.

    Args:
      num_train: number of input training points.
      epsilon: epsilon parameter in (epsilon, delta)-DP.
      delta: delta parameter in (epsilon, delta)-DP.
      num_classes: number of classes.
      input_clipping_norm: l2-norm according to which input points are clipped.

    Returns:
      l2-regularization coefficient and variance of Gaussian noise added in
      Algorithm 1 of Kifer et al.
    """
    # zeta is an upper bound on the l2-norm of the loss function gradient.
    zeta = input_clipping_norm
    # variance is based on line 5 from Algorithm 1 of Kifer et al. (page 6):
    variance = zeta * zeta * (8 * np.log(2 / delta) + 4 * epsilon) / (
        epsilon * epsilon)
    # lambda_coefficient is an upper bound on the spectral norm of the Hessian
    # of the loss function.
    lambda_coefficient = math.sqrt(2 * num_classes) * (input_clipping_norm**
                                                       2) / 4
    l2_regularizer = lambda_coefficient / (epsilon * num_train)
    return (l2_regularizer, variance)


def logistic_objective_perturbation(train_dataset: datasets.RegressionDataset,
                                    test_dataset: datasets.RegressionDataset,
                                    epsilon: float, delta: float, epochs: int,
                                    num_classes: int,
                                    input_clipping_norm: float) -> List[float]:
  """Trains and validates differentially private logistic regression model.

    The training is based on the Algorithm 1 of Kifer et al.

  Args:
    train_dataset: consists of num_train many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    test_dataset: consists of num_test many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    epsilon: epsilon parameter in (epsilon, delta)-DP.
    delta: delta parameter in (epsilon, delta)-DP.
    epochs: number of training epochs.
    num_classes: number of classes.
    input_clipping_norm: l2-norm according to which input points are clipped.

  Returns:
    List of test accuracies (one for each epoch) on test_dataset of model
    trained on train_dataset.
  """
  num_train, dimension = train_dataset.points.shape
  # Normalize each training point (i.e., row of train_dataset.points) to have
  # l2-norm at most input_clipping_norm.
  train_dataset.points = tf.clip_by_norm(train_dataset.points,
                                         input_clipping_norm, [1]).numpy()
  optimizer = 'sgd'
  loss = 'categorical_crossentropy'
  kernel_regularizer = KiferRegularizer(num_train, dimension, epsilon, delta,
                                        num_classes, input_clipping_norm)
  return single_layer_softmax.single_layer_softmax_classifier(
      train_dataset,
      test_dataset,
      epochs,
      num_classes,
      optimizer,
      loss,
      kernel_regularizer=kernel_regularizer)


def compute_dpsgd_noise_multiplier(num_train: int,
                                   epsilon: float,
                                   delta: float,
                                   epochs: int,
                                   batch_size: int,
                                   tolerance: float = 1e-2) -> Optional[float]:
  """Computes the noise multiplier for DP-SGD given privacy parameters.

    The algorithm performs binary search on the values of epsilon.

  Args:
    num_train: number of input training points.
    epsilon: epsilon parameter in (epsilon, delta)-DP.
    delta: delta parameter in (epsilon, delta)-DP.
    epochs: number of training epochs.
    batch_size: the number of examples in each batch of gradient descent.
    tolerance: an upper bound on the absolute difference between the input
      (desired) epsilon and the epsilon value corresponding to the
      noise_multiplier that is output.

  Returns:
    noise_multiplier: the smallest noise multiplier value (within plus or minus
    the given tolerance) for which using DPKerasAdamOptimizer will result in an
    (epsilon, delta)-differentially private trained model.
  """
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  steps = int(math.ceil(epochs * num_train / batch_size))

  def make_event_from_param(noise_multiplier):
    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability=batch_size / num_train,
            event=dp_accounting.GaussianDpEvent(noise_multiplier)), steps)

  return dp_accounting.calibrate_dp_mechanism(
      lambda: dp_accounting.rdp.RdpAccountant(orders),
      make_event_from_param,
      epsilon,
      delta,
      dp_accounting.LowerEndpointAndGuess(0, 1),
      tol=tolerance)


def logistic_dpsgd(train_dataset: datasets.RegressionDataset,
                   test_dataset: datasets.RegressionDataset, epsilon: float,
                   delta: float, epochs: int, num_classes: int, batch_size: int,
                   num_microbatches: int, clipping_norm: float) -> List[float]:
  """Trains and validates private logistic regression model via DP-SGD.

    The training is based on the differentially private stochasstic gradient
    descent algorithm implemented in TensorFlow Privacy.

  Args:
    train_dataset: consists of num_train many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    test_dataset: consists of num_test many labeled examples, where the labels
      are in {0,1,...,num_classes-1}.
    epsilon: epsilon parameter in (epsilon, delta)-DP.
    delta: delta parameter in (epsilon, delta)-DP.
    epochs: number of training epochs.
    num_classes: number of classes.
    batch_size: the number of examples in each batch of gradient descent.
    num_microbatches: the number of microbatches in gradient descent.
    clipping_norm: the gradients will be normalized by DPKerasAdamOptimizer to
      have l2-norm at most clipping_norm.

  Returns:
    List of test accuracies (one for each epoch) on test_dataset of model
    trained on train_dataset.
  """
  num_train = train_dataset.points.shape[0]
  remainder = num_train % batch_size
  if remainder != 0:
    train_dataset.points = train_dataset.points[:-remainder, :]
    train_dataset.labels = train_dataset.labels[:-remainder]
    num_train -= remainder
  noise_multiplier = compute_dpsgd_noise_multiplier(num_train, epsilon, delta,
                                                    epochs, batch_size)
  optimizer = dp_optimizer_keras.DPKerasAdamOptimizer(
      l2_norm_clip=clipping_norm,
      noise_multiplier=noise_multiplier,
      num_microbatches=num_microbatches)
  loss = tf.keras.losses.CategoricalCrossentropy(
      reduction=tf.losses.Reduction.NONE)
  return single_layer_softmax.single_layer_softmax_classifier(
      train_dataset, test_dataset, epochs, num_classes, optimizer, loss,
      batch_size)

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
"""Bolton model for bolton method of differentially private ML"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers
from tensorflow.python.framework import ops as _ops
from privacy.bolton.loss import StrongConvexMixin
from privacy.bolton.optimizer import Bolton

_accepted_distributions = ['laplace']


class BoltonModel(Model):
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

  def __init__(self,
               n_classes,
               # noise_distribution='laplace',
               seed=1,
               dtype=tf.float32
               ):
    """ private constructor.

    Args:
        n_classes: number of output classes to predict.
        epsilon: level of privacy guarantee
        noise_distribution: distribution to pull weight perturbations from
        weights_initializer: initializer for weights
        seed: random seed to use
        dtype: data type to use for tensors
    """

    # if noise_distribution not in _accepted_distributions:
    #   raise ValueError('Detected noise distribution: {0} not one of: {1} valid'
    #                    'distributions'.format(noise_distribution,
    #                                           _accepted_distributions))
    # if epsilon <= 0:
    #   raise ValueError('Detected epsilon: {0}. '
    #                    'Valid range is 0 < epsilon <inf'.format(epsilon))
    # self.epsilon = epsilon
    super(BoltonModel, self).__init__(name='bolton', dynamic=False)
    self.n_classes = n_classes
    self.force = False
    # self.noise_distribution = noise_distribution
    self.seed = seed
    self.__in_fit = False
    self._layers_instantiated = False
    # self._callback = MyCustomCallback()
    self._dtype = dtype

  def call(self, inputs):
    """Forward pass of network

    Args:
        inputs: inputs to neural network

    Returns:

    """
    return self.output_layer(inputs)

  def compile(self,
              optimizer='SGD',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
    """See super class. Default optimizer used in Bolton method is SGD.

    """
    for key, val in StrongConvexMixin.__dict__.items():
      if callable(val) and getattr(loss, key, None) is None:
        raise ValueError("Please ensure you are passing a valid StrongConvex "
                         "loss that has all the required methods "
                         "implemented. "
                         "Required method: {0} not found".format(key))
    if not self._layers_instantiated:  # compile may be called multiple times
      kernel_intiializer = kwargs.get('kernel_initializer',
                                      tf.initializers.GlorotUniform)
      self.output_layer = tf.keras.layers.Dense(
          self.n_classes,
          kernel_regularizer=loss.kernel_regularizer(),
          kernel_initializer=kernel_intiializer(),
      )
      # if we don't do regularization here, we require the user to
      # re-instantiate the model each time they want to change the penalty
      # weighting
      self._layers_instantiated = True
    self.output_layer.kernel_regularizer.l2 = loss.reg_lambda
    if not isinstance(optimizer, Bolton):
      optimizer = optimizers.get(optimizer)
      optimizer = Bolton(optimizer, loss)

    super(BoltonModel, self).compile(optimizer,
                                     loss=loss,
                                     metrics=metrics,
                                     loss_weights=loss_weights,
                                     sample_weight_mode=sample_weight_mode,
                                     weighted_metrics=weighted_metrics,
                                     target_tensors=target_tensors,
                                     distribute=distribute,
                                     **kwargs
                                     )

  # def _post_fit(self, x, n_samples):
  #   """Implements 1-time weight changes needed for Bolton method.
  #   In this case, specifically implements the noise addition
  #   assuming a strongly convex function.
  #
  #   Args:
  #       x: inputs
  #       n_samples: number of samples in the inputs. In case the number
  #       cannot be readily determined by inspecting x.
  #
  #   Returns:
  #
  #   """
  #   data_size = None
  #   if n_samples is not None:
  #     data_size = n_samples
  #   elif hasattr(x, 'shape'):
  #     data_size = x.shape[0]
  #   elif hasattr(x, "__len__"):
  #     data_size = len(x)
  #   elif data_size is None:
  #     if n_samples is None:
  #       raise ValueError("Unable to detect the number of training "
  #                        "samples and n_smaples was None. "
  #                        "either pass a dataset with a .shape or "
  #                        "__len__ attribute or explicitly pass the "
  #                        "number of samples as n_smaples.")
  #   for layer in self.layers:
  #     # layer.kernel = layer.kernel + self._get_noise(
  #     #     data_size
  #     # )
  #     input_dim = layer.kernel.numpy().shape[0]
  #     layer.kernel = layer.kernel + self.optimizer.get_noise(
  #         self.loss,
  #         data_size,
  #         input_dim,
  #         self.n_classes,
  #         self.class_weight
  #     )

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          n_samples=None,
          epsilon=2,
          noise_distribution='laplace',
          **kwargs):
    """Reroutes to super fit with additional Bolton delta-epsilon privacy
    requirements implemented. Note, inputs must be normalized s.t. ||x|| < 1
    Requirements are as follows:
        1. Adds noise to weights after training (output perturbation).
        2. Projects weights to R after each batch
        3. Limits learning rate
        4. Use a strongly convex loss function (see compile)
    See super implementation for more details.

    Args:
        n_samples: the number of individual samples in x.

    Returns:

    """
    self.__in_fit = True
    # cb = [self.optimizer.callbacks]
    # if callbacks is not None:
    #   cb.extend(callbacks)
    # callbacks = cb
    if class_weight is None:
      class_weight = self.calculate_class_weights(class_weight)
    # self.class_weight = class_weight
    with self.optimizer(noise_distribution,
                        epsilon,
                        self.layers,
                        class_weight,
                        n_samples,
                        self.n_classes,
                        ) as optim:
      out = super(BoltonModel, self).fit(x=x,
                                         y=y,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         verbose=verbose,
                                         callbacks=callbacks,
                                         validation_split=validation_split,
                                         validation_data=validation_data,
                                         shuffle=shuffle,
                                         class_weight=class_weight,
                                         sample_weight=sample_weight,
                                         initial_epoch=initial_epoch,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_steps=validation_steps,
                                         validation_freq=validation_freq,
                                         max_queue_size=max_queue_size,
                                         workers=workers,
                                         use_multiprocessing=use_multiprocessing,
                                         **kwargs
                                         )
    return out

  def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0,
                    n_samples=None
                    ):
    """
        This method is the same as fit except for when the passed dataset
        is a generator. See super method and fit for more details.
    Args:
        n_samples: number of individual samples in x

    """
    if class_weight is None:
      class_weight = self.calculate_class_weights(class_weight)
    self.class_weight = class_weight
    out = super(BoltonModel, self).fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch
    )
    if not self.__in_fit:
      self._post_fit(generator, n_samples)
    return out

  def calculate_class_weights(self,
                              class_weights=None,
                              class_counts=None,
                              num_classes=None
                              ):
    """
        Calculates class weighting to be used in training. Can be on
    Args:
        class_weights: str specifying type, array giving weights, or None.
        class_counts: If class_weights is not None, then an array of
                      the number of samples for each class
        num_classes: If class_weights is not None, then the number of
                        classes.
    Returns: class_weights as 1D tensor, to be passed to model's fit method.

    """
    # Value checking
    class_keys = ['balanced']
    is_string = False
    if isinstance(class_weights, str):
      is_string = True
      if class_weights not in class_keys:
        raise ValueError("Detected string class_weights with "
                         "value: {0}, which is not one of {1}."
                         "Please select a valid class_weight type"
                         "or pass an array".format(class_weights,
                                                   class_keys))
      if class_counts is None:
        raise ValueError("Class counts must be provided if using "
                         "class_weights=%s" % class_weights)
      class_counts_shape = tf.Variable(class_counts,
                                       trainable=False,
                                       dtype=self._dtype).shape
      if len(class_counts_shape) != 1:
        raise ValueError('class counts must be a 1D array.'
                         'Detected: {0}'.format(class_counts_shape))
      if num_classes is None:
        raise ValueError("num_classes must be provided if using "
                         "class_weights=%s" % class_weights)
    elif class_weights is not None:
      if num_classes is None:
        raise ValueError("You must pass a value for num_classes if"
                         "creating an array of class_weights")
    # performing class weight calculation
    if class_weights is None:
      class_weights = 1
    elif is_string and class_weights == 'balanced':
      num_samples = sum(class_counts)
      weighted_counts = tf.dtypes.cast(tf.math.multiply(num_classes,
                                                        class_counts,
                                                        ),
                                       self._dtype
                                       )
      class_weights = tf.Variable(num_samples, dtype=self._dtype) / \
                      tf.Variable(weighted_counts, dtype=self._dtype)
    else:
      class_weights = _ops.convert_to_tensor_v2(class_weights)
      if len(class_weights.shape) != 1:
        raise ValueError("Detected class_weights shape: {0} instead of "
                         "1D array".format(class_weights.shape))
      if class_weights.shape[0] != num_classes:
        raise ValueError(
          "Detected array length: {0} instead of: {1}".format(
            class_weights.shape[0],
            num_classes
          )
        )
    return class_weights

  # def _project_weights_to_r(self, r, force=False):
  #   """helper method to normalize the weights to the R-ball.
  #
  #   Args:
  #       r: radius of "R-Ball". Scalar to normalize to.
  #       force: True to normalize regardless of previous weight values.
  #               False to check if weights > R-ball and only normalize then.
  #
  #   Returns:
  #
  #   """
  #   for layer in self.layers:
  #     weight_norm = tf.norm(layer.kernel, axis=0)
  #     if force:
  #       layer.kernel = layer.kernel / (weight_norm / r)
  #     elif tf.reduce_sum(tf.cast(weight_norm > r, dtype=self._dtype)) > 0:
  #       layer.kernel = layer.kernel / (weight_norm / r)

  # def _get_noise(self, distribution, data_size):
  #   """Sample noise to be added to weights for privacy guarantee
  #
  #   Args:
  #       distribution: the distribution type to pull noise from
  #       data_size: the number of samples
  #
  #   Returns: noise in shape of layer's weights to be added to the weights.
  #
  #   """
  #   distribution = distribution.lower()
  #   input_dim = self.layers[0].kernel.numpy().shape[0]
  #   loss = self.loss
  #   if distribution == _accepted_distributions[0]:  # laplace
  #     per_class_epsilon = self.epsilon / (self.n_classes)
  #     l2_sensitivity = (2 *
  #                       loss.lipchitz_constant(self.class_weight)) / \
  #                      (loss.gamma() * data_size)
  #     unit_vector = tf.random.normal(shape=(input_dim, self.n_classes),
  #                                    mean=0,
  #                                    seed=1,
  #                                    stddev=1.0,
  #                                    dtype=self._dtype)
  #     unit_vector = unit_vector / tf.math.sqrt(
  #         tf.reduce_sum(tf.math.square(unit_vector), axis=0)
  #     )
  #
  #     beta = l2_sensitivity / per_class_epsilon
  #     alpha = input_dim  # input_dim
  #     gamma = tf.random.gamma([self.n_classes],
  #                             alpha,
  #                             beta=1 / beta,
  #                             seed=1,
  #                             dtype=self._dtype
  #                             )
  #     return unit_vector * gamma
  #   raise NotImplementedError('Noise distribution: {0} is not '
  #                             'a valid distribution'.format(distribution))


if __name__ == '__main__':
  import tensorflow as tf

  import os
  import time
  import matplotlib.pyplot as plt

  _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

  path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                        origin=_URL,
                                        extract=True)

  PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
  BUFFER_SIZE = 400
  BATCH_SIZE = 1
  IMG_WIDTH = 256
  IMG_HEIGHT = 256


  def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


  inp, re = load(PATH + 'train/100.jpg')
  # casting to int for matplotlib to show the image
  plt.figure()
  plt.imshow(inp / 255.0)
  plt.figure()
  plt.imshow(re / 255.0)


  def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


  def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


  def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


  @tf.function()
  def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
      # random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


  def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


  def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


  train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(1)
  # steps_per_epoch = training_utils.infer_steps_for_dataset(
  #     train_dataset, None, epochs=1, steps_name='steps')

  # for batch in train_dataset:
  #     print(batch[1].shape)
  test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
  # shuffling so that for every epoch a different image is generated
  # to predict and display the progress of our model.
  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  test_dataset = test_dataset.map(load_image_test)
  test_dataset = test_dataset.batch(1)

  be = BoltonModel(3, 2)
  from tensorflow.python.keras.optimizer_v2 import adam
  from privacy.bolton import loss

  test = adam.Adam()
  l = loss.StrongConvexBinaryCrossentropy(1, 2, 1)
  be.compile(test, l)
  print("Eager exeuction: {0}".format(tf.executing_eagerly()))
  be.fit(train_dataset, verbose=0, steps_per_epoch=1, n_samples=1)

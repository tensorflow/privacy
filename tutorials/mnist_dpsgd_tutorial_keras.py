# Copyright 2019, The TensorFlow Authors.
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
"""Training a CNN on MNIST with Keras and the DP SGD optimizer.

**************************** PLEASE READ ME ************************************

A modification to Keras needed for this tutorial to work as it is currently
written is *being* pushed. While this modification is in the works, you can
make this tutorial work by making the following change to the TensorFlow source
code (disabling the reduction of the loss used to compile a model):

Diff for file: tensorflow/python/keras/engine/training_utils.py

```
+ from tensorflow.python.ops.losses import losses_impl

  def get_loss_function():

    ...

-   return losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__)
+   return losses.LossFunctionWrapper(loss_fn,
+                                     name=loss_fn.__name__,
+                                     reduction=losses_impl.Reduction.NONE)
```

This allows the DP-SGD optimizer to have access to the loss defined per
example rather than the mean of the loss for the entire minibatch. This is
needed to compute gradients for each microbatch contained in a minibatch.

**************************** END OF PLEASE READ ME *****************************

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer
from privacy.optimizers.gaussian_query import GaussianAverageQuery

# Compatibility with tf 1 and 2 APIs
try:
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
except:  # pylint: disable=bare-except
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

tf.flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, '
                        'train with vanilla SGD.')
tf.flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 1.1,
                      'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('batch_size', 250, 'Batch size')
tf.flags.DEFINE_integer('epochs', 60, 'Number of epochs')
tf.flags.DEFINE_integer('microbatches', 250, 'Number of microbatches '
                        '(must evenly divide batch_size)')
tf.flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = tf.flags.FLAGS


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  if FLAGS.dpsgd:
    dp_average_query = GaussianAverageQuery(
        FLAGS.l2_norm_clip,
        FLAGS.l2_norm_clip * FLAGS.noise_multiplier,
        FLAGS.microbatches)
    optimizer = DPGradientDescentOptimizer(
        dp_average_query,
        FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate,
        unroll_microbatches=True)
  else:
    optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

  def keras_loss_fn(labels, logits):
    """This removes the mandatory named arguments for this loss fn."""
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                      logits=logits)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=keras_loss_fn, metrics=['accuracy'])

  # Train model with Keras
  model.fit(train_data, train_labels,
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            batch_size=FLAGS.batch_size)

  # Compute the privacy budget expended.
  if FLAGS.noise_multiplier == 0.0:
    print('Trained with vanilla non-private SGD optimizer')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=(FLAGS.epochs * 60000 // FLAGS.batch_size),
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
  print('For delta=1e-5, the current epsilon is: %.2f' % eps)

if __name__ == '__main__':
  tf.app.run()

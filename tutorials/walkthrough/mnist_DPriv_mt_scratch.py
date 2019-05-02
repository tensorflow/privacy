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


"""Scratchpad for training a CNN on MNIST with DPSGD."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(0, '/Users/trog/Documents/Spyder/privacy')
import numpy as np
import tensorflow as tf
from privacy.optimizers import dp_optimizer


learning_rate=0.25
noise_multiplier=1.3
l2_norm_clip=1.5
batch_size=256
epochs=2
num_microbatches=256
try:
    tf.flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate for training')
    tf.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    tf.flags.DEFINE_integer('epochs', epochs, 'Number of epochs')
    tf.flags.DEFINE_float("noise_multiplier",noise_multiplier, "Noise Mult")
    tf.flags.DEFINE_integer("num_microbatches",num_microbatches, "num microbatches")
    tf.flags.DEFINE_float("l2_norm_clip" , l2_norm_clip, "L2 clipping ")
except :
    print("Duplicate Flags Skip...")
    
FLAGS = tf.flags.FLAGS


class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to print current value of epsilon after an epoch."""

  def __init__(self, ledger):
    """Initalizes the EpsilonPrintingTrainingHook.

    Args:
      ledger: The privacy ledger.
    """
    self._samples, self._queries = ledger.get_unformatted_ledger()

  def end(self, session):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    samples = session.run(self._samples)
    queries = session.run(self._queries)
    formatted_ledger = privacy_ledger.format_ledger(samples, queries)
    rdp = compute_rdp_from_ledger(formatted_ledger, orders)
    eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)

def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""

  # Define CNN architecture using tf.keras.layers.
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)

  # Calculate loss as a vector and as its average across minibatch.
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                               logits=logits)
  scalar_loss = tf.reduce_mean(vector_loss)
  

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("Train data mode")
    #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    optimizer=dp_optimizer.DPGradientDescentGaussianOptimizer(l2_norm_clip=FLAGS.l2_norm_clip,
                                                                        noise_multiplier=FLAGS.noise_multiplier,
                                                                        num_microbatches=FLAGS.num_microbatches,
                                                                        learning_rate=FLAGS.learning_rate,
                                                                        population_size=60000,
                                                                        training_hooks = [EpsilonPrintingTrainingHook(ledger)])
    
    opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    return( tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      train_op=train_op))

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    print("Evaluate data mode")

    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1))
    }
        
    return (tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops))
  
  predicted_classes = tf.argmax(logits, 1)      
  if mode == tf.estimator.ModeKeys.PREDICT:
      print("predicting data mode")
      predictions={
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
      predictions.values()
    
      return(tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions))

def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  assert train_labels.ndim == 1
  assert test_labels.ndim == 1

  return train_data, train_labels, test_data, test_labels
#
#
#def main(unused_argv):
#  tf.logging.set_verbosity(tf.logging.INFO)
#
#  # Load training and test data.
#  train_data, train_labels, test_data, test_labels = load_mnist()
#
#  # Instantiate the tf.Estimator.
#  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
#
#  # Create tf.Estimator input functions for the training and test data.
#  train_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={'x': train_data},
#      y=train_labels,
#      batch_size=FLAGS.batch_size,
#      num_epochs=FLAGS.epochs,
#      shuffle=True)
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={'x': test_data},
#      y=test_labels,
#      num_epochs=1,
#      shuffle=False)
#
#  # Training loop.
#  steps_per_epoch = 60000 // FLAGS.batch_size
#  for epoch in range(1, FLAGS.epochs + 1):
#    # Train the model for one epoch.
#    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
#
#    # Evaluate the model and print results
#    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#    test_accuracy = eval_results['accuracy']
#    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))
  

tf.logging.set_verbosity(tf.logging.INFO)

  # Load training and test data.
train_data, train_labels, test_data, test_labels = load_mnist()

  # Instantiate the tf.Estimator.
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)

  # Create tf.Estimator input functions for the training and test data.
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=True)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)

pred_input_fn = tf.estimator.inputs.numpy_input_fn( {'x':test_data[1]},shuffle=False,num_epochs=1)

  # Training loop.
steps_per_epoch = 60000 // FLAGS.batch_size
for epoch in range(1, FLAGS.epochs + 1):
    # Train the model for one epoch.
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))
    
#    predict_results = mnist_classifier.predict( input_fn=pred_input_fn)
#
#    for result in predict_results:
#       print ('result: {}'.format(result))

#if __name__ == '__main__':
  #tf.app.run()
#labels = set(train_labels)
#predVal = mnist_classifier.predict(input_fn=pred_input_fn, predict_keys = labels)
#predict_results = mnist_classifier.predict( input_fn=lambda:eval_input_fn(pred_input_fn, None, 1))



#tf.logging.info("Predictions:") for pred_dict, expec in zip(predict_results, expected): template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

#class_id = pred_dict['class_ids'][0]
#probability = pred_dict['probabilities'][class_id]
#print(template.format(CLASSES[class_id],100 * probability, expec))
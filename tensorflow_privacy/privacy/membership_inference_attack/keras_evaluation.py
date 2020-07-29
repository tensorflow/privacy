# Copyright 2020, The TensorFlow Authors.
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

# Lint as: python3
"""A callback and a function in keras for membership inference attack."""

from absl import logging

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.utils import log_loss
from tensorflow_privacy.privacy.membership_inference_attack.utils import write_to_tensorboard


def calculate_losses(model, data, labels):
  """Calculate losses of model prediction on data, provided true labels.

  Args:
    model: model to make prediction
    data: samples
    labels: true labels of samples (integer valued)

  Returns:
    preds: probability vector of each sample
    loss: cross entropy loss of each sample
  """
  pred = model.predict(data)
  loss = log_loss(labels, pred)
  return pred, loss


class MembershipInferenceCallback(tf.keras.callbacks.Callback):
  """Callback to perform membership inference attack on epoch end."""

  def __init__(self, in_train, out_train, attack_classifiers,
               tensorboard_dir=None):
    """Initalizes the callback.

    Args:
      in_train: (in_training samples, in_training labels)
      out_train: (out_training samples, out_training labels)
      attack_classifiers: a list of classifiers to be used by attacker, must be
        a subset of ['lr', 'mlp', 'rf', 'knn']
      tensorboard_dir: directory for tensorboard summary
    """
    self._in_train_data, self._in_train_labels = in_train
    self._out_train_data, self._out_train_labels = out_train
    self._attack_classifiers = attack_classifiers
    # Setup tensorboard writer if tensorboard_dir is specified
    if tensorboard_dir:
      with tf.Graph().as_default():
        self._writer = tf.summary.FileWriter(tensorboard_dir)
      logging.info('Will write to tensorboard.')
    else:
      self._writer = None

  def on_epoch_end(self, epoch, logs=None):
    results = run_attack_on_keras_model(
        self.model,
        (self._in_train_data, self._in_train_labels),
        (self._out_train_data, self._out_train_labels),
        self._attack_classifiers)
    print('all_thresh_loss_advantage', results['all_thresh_loss_advantage'])
    logging.info(results)

    # Write to tensorboard if tensorboard_dir is specified
    write_to_tensorboard(self._writer, ['attack advantage'],
                         [results['all_thresh_loss_advantage']], epoch)


def run_attack_on_keras_model(model, in_train, out_train, attack_classifiers):
  """Performs the attack on a trained model.

  Args:
    model: model to be tested
    in_train: a (in_training samples, in_training labels) tuple
    out_train: a (out_training samples, out_training labels) tuple
    attack_classifiers: a list of classifiers to be used by attacker, must be
      a subset of ['lr', 'mlp', 'rf', 'knn']
  Returns:
    Results of the attack
  """
  in_train_data, in_train_labels = in_train
  out_train_data, out_train_labels = out_train

  # Compute predictions and losses
  in_train_pred, in_train_loss = calculate_losses(model, in_train_data,
                                                  in_train_labels)
  out_train_pred, out_train_loss = calculate_losses(model, out_train_data,
                                                    out_train_labels)
  results = mia.run_all_attacks(in_train_loss, out_train_loss,
                                in_train_pred, out_train_pred,
                                in_train_labels, out_train_labels,
                                attack_classifiers=attack_classifiers)
  return results


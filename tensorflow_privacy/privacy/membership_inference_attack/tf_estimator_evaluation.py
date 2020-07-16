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
"""A hook and a function in tf estimator for membership inference attack."""

from absl import logging

import numpy as np

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.utils import log_loss


def calculate_losses(estimator, input_fn, labels):
  """Get predictions and losses for samples.

  The assumptions are 1) the loss is cross-entropy loss, and 2) user have
  specified prediction mode to return predictions, e.g.,
  when mode == tf.estimator.ModeKeys.PREDICT, the model function returns
  tf.estimator.EstimatorSpec(mode=mode, predictions=tf.nn.softmax(logits)).

  Args:
    estimator: model to make prediction
    input_fn: input function to be used in estimator.predict
    labels: true labels of samples

  Returns:
    preds: probability vector of each sample
    loss: cross entropy loss of each sample
  """
  pred = np.array(list(estimator.predict(input_fn=input_fn)))
  loss = log_loss(labels, pred)
  return pred, loss


class MembershipInferenceTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to perform membership inference attack after an epoch."""

  def __init__(self,
               estimator,
               in_train,
               out_train,
               input_fn_constructor,
               attack_classifiers,
               writer=None):
    """Initalizes the hook.

    Args:
      estimator: model to be tested
      in_train: (in_training samples, in_training labels)
      out_train: (out_training samples, out_training labels)
      input_fn_constructor: a function that receives sample, label and construct
        the input_fn for model prediction
      attack_classifiers: a list of classifiers to be used by attacker, must be
        a subset of ['lr', 'mlp', 'rf', 'knn']
      writer: summary writer for tensorboard
    """
    in_train_data, self._in_train_labels = in_train
    out_train_data, self._out_train_labels = out_train

    # Define the input functions for both in and out-training samples.
    self._in_train_input_fn = input_fn_constructor(in_train_data,
                                                   self._in_train_labels)
    self._out_train_input_fn = input_fn_constructor(out_train_data,
                                                    self._out_train_labels)
    self._estimator = estimator
    self._attack_classifiers = attack_classifiers
    self._writer = writer
    if self._writer:
      logging.info('Will write to tensorboard.')

  def end(self, session):
    results = run_attack_helper(self._estimator,
                                self._in_train_input_fn,
                                self._out_train_input_fn,
                                self._in_train_labels, self._out_train_labels,
                                self._attack_classifiers)
    print('all_thresh_loss_advantage', results['all_thresh_loss_advantage'])
    logging.info(results)

    if self._writer:
      summary = tf.Summary()
      summary.value.add(tag='attack advantage',
                        simple_value=results['all_thresh_loss_advantage'])
      global_step = self._estimator.get_variable_value('global_step')
      self._writer.add_summary(summary, global_step)
      self._writer.flush()


def run_attack_on_tf_estimator_model(estimator, in_train, out_train,
                                     input_fn_constructor, attack_classifiers):
  """A function to perform the attack in the end of training.

  Args:
    estimator: model to be tested
    in_train: (in_training samples, in_training labels)
    out_train: (out_training samples, out_training labels)
    input_fn_constructor: a function that receives sample, label and construct
      the input_fn for model prediction
    attack_classifiers: a list of classifiers to be used by attacker, must be
      a subset of ['lr', 'mlp', 'rf', 'knn']
  Returns:
    Results of the attack
  """
  in_train_data, in_train_labels = in_train
  out_train_data, out_train_labels = out_train

  # Define the input functions for both in and out-training samples.
  in_train_input_fn = input_fn_constructor(in_train_data, in_train_labels)
  out_train_input_fn = input_fn_constructor(out_train_data, out_train_labels)

  # Call the helper to run the attack.
  results = run_attack_helper(estimator,
                              in_train_input_fn, out_train_input_fn,
                              in_train_labels, out_train_labels,
                              attack_classifiers)
  print('all_thresh_loss_advantage', results['all_thresh_loss_advantage'])
  logging.info('End of training attack:')
  logging.info(results)
  return results


def run_attack_helper(estimator,
                      in_train_input_fn, out_train_input_fn,
                      in_train_labels, out_train_labels,
                      attack_classifiers):
  """A helper function to perform attack.

  Args:
    estimator: model to be tested
    in_train_input_fn: input_fn for in training data
    out_train_input_fn: input_fn for out of training data
    in_train_labels: in training labels
    out_train_labels: out of training labels
    attack_classifiers: a list of classifiers to be used by attacker, must be
      a subset of ['lr', 'mlp', 'rf', 'knn']
  Returns:
    Results of the attack
  """
  # Compute predictions and losses
  in_train_pred, in_train_loss = calculate_losses(estimator,
                                                  in_train_input_fn,
                                                  in_train_labels)
  out_train_pred, out_train_loss = calculate_losses(estimator,
                                                    out_train_input_fn,
                                                    out_train_labels)
  results = mia.run_all_attacks(in_train_loss, out_train_loss,
                                in_train_pred, out_train_pred,
                                in_train_labels, out_train_labels,
                                attack_classifiers=attack_classifiers)
  return results


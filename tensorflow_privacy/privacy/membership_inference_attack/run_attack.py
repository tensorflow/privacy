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
r"""This module contains code to run attacks on previous model outputs.

Provided a path to a dataset of model outputs (logits, output probabilities,
losses, labels, predictions, membership indicators (is train or not)), we train
supervised binary classifiers using variable sets of features to distinguish
training from testing examples. We also provide threshold attacks, i.e., simply
thresholing losses or the maximum probability/logit to obtain binary
predictions.

The input data is assumed to be a tf.example proto stored with RecordIO (.rio).
For example, outputs in an accepted format are typically produced by the
`extract` script in the `extract` directory.

We run various attacks on each of the full datasets, split by class, split by
percentile of the most certain prediction and only on misclassified examples and
record the area under the receiver operator curve as well as the attack
advantage (i.e., max |tpr - fpr|) as vulnerability metrics. For all metrics
recorded, see the doc string of `membership_inference_attack.all_attacks`.
In addition, we record the overall training and test accuracy and loss of the
original image classifier. All these results are collected in a single
dictionary with descriptive keys. If there exist multiple model checkpoints (at
different training epochs), the results for each checkpoint are concatenated,
such that the dictionary keys stay the same, but the values contain arrays (the
size being the number of checkpoints). This overall result dicitonary is then
stored as a binary (and compressed) numpy file: .npz.
This file is stored either in the provided output path. If that is the empty
string, it is stored on the same level as the inputdir with the chosen name.
Using `attack_results.npz` by default.
"""

# Example usage:

python run_attack.py --dataset=cifar10 --inputdir="attack_data"
The results are then stored at ./attack_data

import io
import os
import re

from typing import Text, Dict

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.google as tf
import tensorflow_datasets as tfds

from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack import utils

from glob import glob

Result = Dict[Text, np.ndarray]

FLAGS = flags.FLAGS

flags.DEFINE_float('test_size', 0.2,
                   'Fraction of attack data used for the test set.')
flags.DEFINE_string('dataset', 'cifar10', 'The dataset to use.')
flags.DEFINE_string(
    'output', '', 'The path where to store the results. '
    'If empty string, store on same level as `inputdir` using '
    'the name specified in the result_name flag.')
flags.DEFINE_string('result_name', 'attack_results.npz',
                    'The name of the output npz file with the attack results.')
flags.DEFINE_string(
    'inputdir',
    'attack_data',
    'The input directory containing the attack datasets.')
flags.DEFINE_integer('seed', 43, 'Random seed to ensure same data splits.')

# ------------------------------------------------------------------------------
#  Load and select features for attacks
# ------------------------------------------------------------------------------


def load_all_features(data_path: Text) -> Result:
  """Extract the selected features from a given dataset."""
  if FLAGS.dataset == 'cifar100':
    num_classes = 100
  elif FLAGS.dataset in ['cifar10', 'mnist']:
    num_classes = 10
  else:
    raise ValueError(f'Unknown dataset {FLAGS.dataset}')

  features = {
      'logits': tf.FixedLenFeature((num_classes,), tf.float32),
      'prob': tf.FixedLenFeature((num_classes,), tf.float32),
      'loss': tf.FixedLenFeature([], tf.float32),
      'is_train': tf.FixedLenFeature([], tf.int64),
      'label': tf.FixedLenFeature([], tf.int64),
      'prediction': tf.FixedLenFeature([], tf.int64),
  }

  dataset = tf.data.RecordIODataset(data_path)

  results = {k: [] for k in features}
  ds = dataset.map(lambda x: tf.parse_single_example(x, features))
  for example in tfds.as_numpy(ds):
    for k in results:
      results[k].append(example[k])
  return utils.to_numpy(results)


# ------------------------------------------------------------------------------
#  Run attacks
# ------------------------------------------------------------------------------


def run_all_attacks(data_path: Text):
  """Train all possible attacks on the data at the given path."""
  logging.info('Load all features from %s...', data_path)
  features = load_all_features(data_path)

  for k, v in features.items():
    logging.info('%s: %s', k, v.shape)

  logging.info('Compute original train/test accuracy and loss...')
  train_idx = features['is_train'] == 1
  test_idx = np.logical_not(train_idx)
  correct = features['label'] == features['prediction']
  result = {
      'original_train_loss': np.mean(features['loss'][train_idx]),
      'original_test_loss': np.mean(features['loss'][test_idx]),
      'original_train_acc': np.mean(correct[train_idx]),
      'original_test_acc': np.mean(correct[test_idx]),
  }

  result.update(
      mia.run_all_attacks(
          loss_train=features['loss'][train_idx],
          loss_test=features['loss'][test_idx],
          logits_train=features['logits'][train_idx],
          logits_test=features['logits'][test_idx],
          labels_train=features['label'][train_idx],
          labels_test=features['label'][test_idx],
          attack_classifiers=('lr', 'mlp', 'rf', 'knn'),
          decimals=None))
  result = utils.ensure_1d(result)

  logging.info('Finished training and evaluating attacks.')
  return result


def attacking():
  """Load data and model and extract relevant outputs."""
  # ---------- Set result path ----------
  if FLAGS.output:
    resultpath = FLAGS.output
  else:
    resultdir = FLAGS.inputdir
    if resultdir[-1] == '/':
      resultdir = resultdir[:-1]
    resultdir = '/'.join(resultdir.split('/')[:-1])
    resultpath = os.path.join(resultdir, FLAGS.result_name)

  # ---------- Glob attack training sets ----------
  logging.info('Glob attack data paths...')
  data_paths = sorted(glob(os.path.join(FLAGS.inputdir, '*')))
  logging.info('Found %d data paths', len(data_paths))

  # ---------- Iterate over attack dataset and train attacks ----------
  epochs = []
  results = []
  for i, datapath in enumerate(data_paths):
    logging.info('=' * 80)
    logging.info('Attack model %d / %d', i + 1, len(data_paths))
    logging.info('=' * 80)
    basename = os.path.basename(datapath)
    found_ints = re.findall(r'(\d+)', basename)
    if len(found_ints) == 1:
      epoch = int(found_ints[0])
      logging.info('Found integer %d in pathname, interpret as epoch', epoch)
    else:
      epoch = np.nan
    tmp_res = run_all_attacks(datapath)
    if tmp_res is not None:
      results.append(tmp_res)
      epochs.append(epoch)

  # ---------- Aggregate and save results ----------
  logging.info('Aggregate and combine all results over epochs...')
  results = utils.merge_dictionaries(results)
  results['epochs'] = np.array(epochs)
  logging.info('Store aggregate results at %s.', resultpath)
  with open(resultpath, 'wb') as fp:
    io_buffer = io.BytesIO()
    np.savez(io_buffer, **results)
    fp.write(io_buffer.getvalue())

  logging.info('Finished attacks.')


def main(argv):
  del argv  # Unused
  attacking()


if __name__ == '__main__':
  app.run(main)

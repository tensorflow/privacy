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
"""Code that runs membership inference attacks based on the model outputs.

Warning: This file belongs to the old API for membership inference attacks. This
file will be removed soon. membership_inference_attack_new.py contains the new
API.
"""

import collections
import io
import os
import re

from typing import Text, Dict, Iterable, Tuple, Union, Any

from absl import logging
import numpy as np
from scipy import special

from tensorflow_privacy.privacy.membership_inference_attack import plotting
from tensorflow_privacy.privacy.membership_inference_attack import trained_attack_models
from tensorflow_privacy.privacy.membership_inference_attack import utils

from os import mkdir

ArrayDict = Dict[Text, np.ndarray]
FloatDict = Dict[Text, float]
AnyDict = Dict[Text, Any]
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
MetricNames = Union[Text, Iterable[Text]]


def _get_vulnerabilities(result: ArrayDict, metrics: MetricNames) -> FloatDict:
  """Gets the vulnerabilities according to the chosen metrics for all attacks."""
  vulns = {}
  if isinstance(metrics, str):
    metrics = [metrics]
  for k in result:
    for metric in metrics:
      if k.endswith(metric.lower()) or k.endswith('n_examples'):
        vulns[k] = float(result[k])
  return vulns


def _get_maximum_vulnerability(
    attack_result: FloatDict,
    metrics: MetricNames,
    filterby: Text = '') -> Dict[Text, Dict[Text, Union[Text, float]]]:
  """Returns the worst vulnerability according to the chosen metrics of all attacks."""
  vulns = {}
  if isinstance(metrics, str):
    metrics = [metrics]
  for metric in metrics:
    best_attack_value = -np.inf
    for k in attack_result:
      if (k.startswith(filterby.lower()) and k.endswith(metric.lower()) and
          'train' not in k):
        if float(attack_result[k]) > best_attack_value:
          best_attack_value = attack_result[k]
          best_attacker = k
    if best_attack_value > -np.inf:
      newkey = filterby + '-' + metric if filterby else metric
      vulns[newkey] = {'value': best_attack_value, 'attacker': best_attacker}
  return vulns


def _get_maximum_class_gap_or_none(result: FloatDict,
                                   metrics: MetricNames) -> FloatDict:
  """Returns the biggest and smallest vulnerability and the gap across classes."""
  gaps = {}
  if isinstance(metrics, str):
    metrics = [metrics]
  for metric in metrics:
    hi = -np.inf
    lo = np.inf
    hi_idx, lo_idx = -1, -1
    for k in result:
      if (k.startswith('class') and k.endswith(metric.lower()) and
          'train' not in k):
        if float(result[k]) > hi:
          hi = float(result[k])
          hi_idx = int(re.findall(r'class_(\d+)_', k)[0])
        if float(result[k]) < lo:
          lo = float(result[k])
          lo_idx = int(re.findall(r'class_(\d+)_', k)[0])
    if lo - hi < np.inf:
      gaps['max_class_gap_' + metric] = hi - lo
      gaps[f'class_{hi_idx}_' + metric] = hi
      gaps[f'class_{lo_idx}_' + metric] = lo
      gaps['max_vuln_class_' + metric] = hi_idx
      gaps['min_vuln_class_' + metric] = lo_idx
  return gaps


# ------------------------------------------------------------------------------
#  Attacks
# ------------------------------------------------------------------------------


def _run_threshold_loss_attack(features: ArrayDict,
                               figure_file_prefix: Text = '',
                               figure_directory: Text = None) -> ArrayDict:
  """Runs the threshold attack on the loss."""
  logging.info('Run threshold attack on loss...')
  is_train = features['is_train']
  attack_prefix = 'thresh_loss'
  tmp_results = utils.compute_performance_metrics(is_train, -features['loss'])
  if figure_directory is not None:
    figpath = os.path.join(figure_directory,
                           figure_file_prefix + attack_prefix + '.png')
    plotting.save_plot(
        plotting.plot_curve_with_area(
            tmp_results['fpr'], tmp_results['tpr'], xlabel='FPR', ylabel='TPR'),
        figpath)
    figpath = os.path.join(figure_directory,
                           figure_file_prefix + attack_prefix + '_hist.png')
    plotting.save_plot(
        plotting.plot_histograms(
            features['loss'][is_train == 1],
            features['loss'][is_train == 0],
            xlabel='loss'), figpath)
  return utils.prepend_to_keys(tmp_results, attack_prefix + '_')


def _run_threshold_attack_maxlogit(features: ArrayDict,
                                   figure_file_prefix: Text = '',
                                   figure_directory: Text = None) -> ArrayDict:
  """Runs the threshold attack on the maximum logit."""
  is_train = features['is_train']
  preds = np.max(features['logits'], axis=-1)
  tmp_results = utils.compute_performance_metrics(is_train, preds)
  attack_prefix = 'thresh_maxlogit'
  if figure_directory is not None:
    figpath = os.path.join(figure_directory,
                           figure_file_prefix + attack_prefix + '.png')
    plotting.save_plot(
        plotting.plot_curve_with_area(
            tmp_results['fpr'], tmp_results['tpr'], xlabel='FPR', ylabel='TPR'),
        figpath)
    figpath = os.path.join(figure_directory,
                           figure_file_prefix + attack_prefix + '_hist.png')
    plotting.save_plot(
        plotting.plot_histograms(
            preds[is_train == 1], preds[is_train == 0], xlabel='loss'), figpath)
  return utils.prepend_to_keys(tmp_results, attack_prefix + '_')


def _run_trained_attack(attack_classifier: Text,
                        data: Dataset,
                        attack_prefix: Text,
                        figure_file_prefix: Text = '',
                        figure_directory: Text = None) -> ArrayDict:
  """Train a classifier for attack and evaluate it."""
  # Train the attack classifier
  (x_train, y_train), (x_test, y_test) = data
  clf_model = trained_attack_models.choose_model(attack_classifier)
  clf_model.fit(x_train, y_train)

  # Calculate training set metrics
  pred_train = clf_model.predict_proba(x_train)[:, clf_model.classes_ == 1]
  results = utils.prepend_to_keys(
      utils.compute_performance_metrics(y_train, pred_train),
      attack_prefix + 'train_')

  # Calculate test set metrics
  pred_test = clf_model.predict_proba(x_test)[:, clf_model.classes_ == 1]
  results.update(
      utils.prepend_to_keys(
          utils.compute_performance_metrics(y_test, pred_test),
          attack_prefix + 'test_'))

  if figure_directory is not None:
    figpath = os.path.join(figure_directory,
                           figure_file_prefix + attack_prefix[:-1] + '.png')
    plotting.save_plot(
        plotting.plot_curve_with_area(
            results[attack_prefix + 'test_fpr'],
            results[attack_prefix + 'test_tpr'],
            xlabel='FPR',
            ylabel='TPR'), figpath)
  return results


def _run_attacks_and_plot(features: ArrayDict,
                          attacks: Iterable[Text],
                          attack_classifiers: Iterable[Text],
                          balance: bool,
                          test_size: float,
                          random_state: int,
                          figure_file_prefix: Text = '',
                          figure_directory: Text = None) -> ArrayDict:
  """Runs the specified attacks on the provided data."""
  if balance:
    try:
      features = utils.subsample_to_balance(features, random_state)
    except RuntimeError:
      logging.info('Not enough remaining data for attack: Empty results.')
      return {}

  result = {}
  # -------------------- Simple threshold attacks
  if 'thresh_loss' in attacks:
    result.update(
        _run_threshold_loss_attack(features, figure_file_prefix,
                                   figure_directory))

  if 'thresh_maxlogit' in attacks:
    result.update(
        _run_threshold_attack_maxlogit(features, figure_file_prefix,
                                       figure_directory))

  # -------------------- Run learned attacks
  # TODO(b/157632603): Add a prefix (for example 'trained_') for attacks which
  # use classifiers to distinguish from threshould attacks.
  if 'logits' in attacks:
    data = utils.get_train_test_split(
        features, add_loss=False, test_size=test_size)
    for clf in attack_classifiers:
      logging.info('Train %s on %d logits', clf, data[0][0].shape[1])
      attack_prefix = f'{clf}_logits_'
      result.update(
          _run_trained_attack(clf, data, attack_prefix, figure_file_prefix,
                              figure_directory))

    if 'logits_loss' in attacks:
      data = utils.get_train_test_split(
          features, add_loss=True, test_size=test_size)
      for clf in attack_classifiers:
        logging.info('Train %s on %d logits + loss', clf, data[0][0].shape[1])
        attack_prefix = f'{clf}_logits_loss_'
        result.update(
            _run_trained_attack(clf, data, attack_prefix, figure_file_prefix,
                                figure_directory))
  return result


def run_attack(loss_train: np.ndarray = None,
               loss_test: np.ndarray = None,
               logits_train: np.ndarray = None,
               logits_test: np.ndarray = None,
               labels_train: np.ndarray = None,
               labels_test: np.ndarray = None,
               attack_classifiers: Iterable[Text] = None,
               only_misclassified: bool = False,
               by_class: Union[bool, Iterable[int], int] = False,
               by_percentile: Union[bool, Iterable[int], int] = False,
               figure_directory: Text = None,
               output_directory: Text = None,
               metric: MetricNames = 'auc',
               balance: bool = True,
               test_size: float = 0.2,
               random_state: int = 0) -> FloatDict:
  """Run membership inference attack(s).

  Based only on specific outputs of a machine learning model on some examples
  used for training (train) and some examples not used for training (test), run
  membership inference attacks that try to discriminate training from test
  inputs based only on the model outputs.
  While all inputs are optional, at least one train/test pair is required to run
  any attacks (either losses or logits/probabilities).
  Note that one can equally provide output probabilities instead of logits in
  the logits_train / logits_test arguments.

  We measure the vulnerability of the model via the area under the ROC-curve
  (auc) or via max |fpr - tpr| (advantage) of the attack classifier. These
  measures are very closely related and may look almost indistinguishable.

  This function provides relatively fine grained control and outputs detailed
  results. For a higher-level wrapper with sane internal default settings and
  distilled output results, see `run_all_attacks`.

  Via the `figure_directory` argument and the `output_directory` argument more
  detailed information as well as roc-curve plots can optionally be stored to
  disk.

  If `loss_train` and `loss_test` are provided we run:
    - simple threshold attack on the loss

  If `logits_train` and `logits_test` are provided we run:
    - simple threshold attack on the top logit
    - if `attack_classifiers` is not None and no losses are provided: train the
       specified classifiers on the top 10 logits (or all logits if there are
       less than 10)
    - if `attack_classifiers` is not None and losses are provided: train the
       specified classifiers on the top 10 logits (or all logits if there are
       less than 10) and the loss

  Args:
    loss_train: A 1D array containing the individual scalar losses for examples
      used during training.
    loss_test: A 1D array containing the individual scalar losses for examples
      not used during training.
    logits_train: A 2D array (n_train, n_classes) of the individual logits or
      output probabilities of examples used during training.
    logits_test: A 2D array (n_test, n_classes) of the individual logits or
      output probabilities of examples not used during training.
    labels_train: The true labels of the training examples. Labels are only
      needed when `by_class` is specified (i.e., not False).
    labels_test: The true labels of the test examples. Labels are only needed
      when `by_class` is specified (i.e., not False).
    attack_classifiers: Attack classifiers to train beyond simple thresholding
      that require training a simple binary ML classifier. This argument is
      ignored if logits are not provided. Classifiers can be 'lr' for logistic
      regression, 'mlp' for multi-layered perceptron, 'rf' for random forests,
      or 'knn' for k-nearest-neighbors. If 'None', don't train classifiers
      beyond simple thresholding.
    only_misclassified: Run and evaluate attacks only on misclassified examples.
      Must specify `labels_train`, `labels_test`, `logits_train` and
      `logits_test` to use this. If this is True, `by_class` and `by_percentile`
      are ignored.
    by_class: This argument determines whether attacks are run on the entire
      data, or on examples grouped by their class label. If `True`, all attacks
      are run separately for each class. If `by_class` is a single integer, run
      attacks for this class only. If `by_class` is an iterable of integers, run
      all attacks for each of the specified class labels separately. Only used
      if `labels_train` and `labels_test` are specified. If `by_class` is
      specified (not False), `by_percentile` is ignored. Ignored if
      `only_misclassified` is True.
    by_percentile: This argument determines whether attacks are run on the
      entire data, or separately for examples where the most likely class
      prediction is within a given percentile of all maximum predicitons. If
      `True`, all attacks are run separately for the examples with max
      probabilities within the ten deciles. If `by_precentile` is a single int
      between 0 and 100, run attacks only for examples with confidence within
      this percentile. If `by_percentile` is an iterable of ints between 0 and
      100, run all attacks for each of the specified percentiles separately.
      Ignored if `by_class` is specified. Ignored if `logits_train` and
      `logits_test` are not specified. Ignored if `only_misclassified` is True.
    figure_directory: Where to store ROC-curve plots and histograms. If `None`,
      don't create plots.
    output_directory: Where to store detailed result data for all run attacks.
      If `None`, don't store detailed result data.
    metric: Available vulnerability metrics are 'auc' or 'advantage' for the
      area under the ROC curve or the advantage (max |tpr - fpr|). Specify
      either one of them or both.
    balance: Whether to use the same number of train and test samples (by
      randomly subsampling whichever happens to be larger).
    test_size: The fraction of the input data to use for the evaluation of
      trained ML attacks. This argument is ignored, if either attack_classifiers
      is None, or no logits are provided.
    random_state: Random seed for reproducibility. Only used if attack models
      are trained.

  Returns:
    results: Dictionary with the chosen vulnerability metric(s) for all ran
      attacks.
  """
  print(
      'Deprecation warning: function run_attack is '
      'deprecated and will be removed soon. '
      'Please use membership_inference_attack_new.run_attacks'
  )
  attacks = []
  features = {}
  # ---------- Check available data ----------
  if ((loss_train is None or loss_test is None) and
      (logits_train is None or logits_test is None)):
    raise ValueError(
        'Need at least train and test for loss or train and test for logits.')

  # ---------- If losses are provided ----------
  if loss_train is not None and loss_test is not None:
    if loss_train.ndim != 1 or loss_test.ndim != 1:
      raise ValueError('Losses must be 1D arrays.')
    features['is_train'] = np.concatenate(
        (np.ones(len(loss_train)), np.zeros(len(loss_test))),
        axis=0).astype(int)
    features['loss'] = np.concatenate((loss_train.ravel(), loss_test.ravel()),
                                      axis=0)
    attacks.append('thresh_loss')

  # ---------- If logits are provided ----------
  if logits_train is not None and logits_test is not None:
    assert logits_train.ndim == 2 and  logits_test.ndim == 2, \
        'Logits must be 2D arrays.'
    assert logits_train.shape[1] == logits_test.shape[1], \
        'Train and test logits must agree along axis 1 (number of classes).'
    if 'is_train' in features:
      assert (loss_train.shape[0] == logits_train.shape[0] and
              loss_test.shape[0] == logits_test.shape[0]), \
          'Number of examples must match between loss and logits.'
    else:
      features['is_train'] = np.concatenate(
          (np.ones(logits_train.shape[0]), np.zeros(logits_test.shape[0])),
          axis=0).astype(int)
    attacks.append('thresh_maxlogit')
    features['logits'] = np.concatenate((logits_train, logits_test), axis=0)
    if attack_classifiers:
      attacks.append('logits')
      if 'loss' in features:
        attacks.append('logits_loss')

  # ---------- If labels are provided ----------
  if labels_train is not None and labels_test is not None:
    if labels_train.ndim != 1 or labels_test.ndim != 1:
      raise ValueError('Labels must be 1D arrays.')
    if 'loss' in features:
      assert (loss_train.shape[0] == labels_train.shape[0] and
              loss_test.shape[0] == labels_test.shape[0]), \
          'Number of examples must match between loss and labels.'
    else:
      assert (logits_train.shape[0] == labels_train.shape[0] and
              logits_test.shape[0] == labels_test.shape[0]), \
          'Number of examples must match between logits and labels.'
    features['label'] = np.concatenate((labels_train, labels_test), axis=0)

  # ---------- Data subsampling or filtering ----------
  filtertype = None
  filtervals = [None]
  if only_misclassified:
    if (labels_train is None or labels_test is None or logits_train is None or
        logits_test is None):
      raise ValueError('Must specify labels_train, labels_test, logits_train, '
                       'and logits_test for the only_misclassified option.')
    filtertype = 'misclassified'
  elif by_class:
    if labels_train is None or labels_test is None:
      raise ValueError('Must specify labels_train and labels_test when using '
                       'the by_class option.')
    if isinstance(by_class, bool):
      filtervals = list(set(labels_train) | set(labels_test))
    elif isinstance(by_class, int):
      filtervals = [by_class]
    elif isinstance(by_class, collections.Iterable):
      filtervals = list(by_class)
    filtertype = 'class'
  elif by_percentile:
    if logits_train is None or logits_test is None:
      raise ValueError('Must specify logits_train and logits_test when using '
                       'the by_percentile option.')
    if isinstance(by_percentile, bool):
      filtervals = list(range(10, 101, 10))
    elif isinstance(by_percentile, int):
      filtervals = [by_percentile]
    elif isinstance(by_percentile, collections.Iterable):
      filtervals = [int(percentile) for percentile in by_percentile]
    filtertype = 'percentile'

  # ---------- Need to create figure directory? ----------
  if figure_directory is not None:
    mkdir(figure_directory)

  # ---------- Actually run attacks and plot if required ----------
  logging.info('Selecting %s with values %s', filtertype, filtervals)
  num = None
  result = {}
  for filterval in filtervals:
    if filtertype is None:
      tmp_features = features
    elif filtertype == 'misclassified':
      idx = features['label'] != np.argmax(features['logits'], axis=-1)
      tmp_features = utils.select_indices(features, idx)
      num = np.sum(idx)
    elif filtertype == 'class':
      idx = features['label'] == filterval
      tmp_features = utils.select_indices(features, idx)
      num = np.sum(idx)
    elif filtertype == 'percentile':
      certainty = np.max(special.softmax(features['logits'], axis=-1), axis=-1)
      idx = certainty <= np.percentile(certainty, filterval)
      tmp_features = utils.select_indices(features, idx)

    prefix = f'{filtertype}_' if filtertype is not None else ''
    prefix += f'{filterval}_' if filterval is not None else ''
    tmp_result = _run_attacks_and_plot(tmp_features, attacks,
                                       attack_classifiers, balance, test_size,
                                       random_state, prefix, figure_directory)
    if num is not None:
      tmp_result['n_examples'] = float(num)
    if tmp_result:
      result.update(utils.prepend_to_keys(tmp_result, prefix))

  # ---------- Store data ----------
  if output_directory is not None:
    mkdir(output_directory)
    resultpath = os.path.join(output_directory, 'attack_results.npz')
    logging.info('Store aggregate results at %s.', resultpath)
    with open(resultpath, 'wb') as fp:
      io_buffer = io.BytesIO()
      np.savez(io_buffer, **result)
      fp.write(io_buffer.getvalue())

  return _get_vulnerabilities(result, metric)


def run_all_attacks(loss_train: np.ndarray = None,
                    loss_test: np.ndarray = None,
                    logits_train: np.ndarray = None,
                    logits_test: np.ndarray = None,
                    labels_train: np.ndarray = None,
                    labels_test: np.ndarray = None,
                    attack_classifiers: Iterable[Text] = ('lr', 'mlp', 'rf',
                                                          'knn'),
                    decimals: Union[int, None] = 4) -> FloatDict:
  """Runs all possible membership inference attacks.

  Check 'run_attack' for detailed information of how attacks are performed
  and evaluated.

  This function internally chooses sane default settings for all attacks and
  returns all possible output combinations.
  For fine grained control and partial attacks, please see `run_attack`.

  Args:
    loss_train: A 1D array containing the individual scalar losses for examples
      used during training.
    loss_test: A 1D array containing the individual scalar losses for examples
      not used during training.
    logits_train: A 2D array (n_train, n_classes) of the individual logits or
      output probabilities of examples used during training.
    logits_test: A 2D array (n_test, n_classes) of the individual logits or
      output probabilities of examples not used during training.
    labels_train: The true labels of the training examples. Labels are only
      needed when `by_class` is specified (i.e., not False).
    labels_test: The true labels of the test examples. Labels are only needed
      when `by_class` is specified (i.e., not False).
    attack_classifiers: Which binary classifiers to train (in addition to simple
      threshold attacks). This can include 'lr' (logistic regression), 'mlp'
      (multi-layered perceptron), 'rf' (random forests), 'knn' (k-nearest
      neighbors), which will be trained with cross validation to determine good
      hyperparameters.
    decimals: Round all float results to this number of decimals. If decimals is
      None, don't round.

  Returns:
    result: dictionary with all attack results
  """
  print(
      'Deprecation warning: function run_all_attacks is '
      'deprecated and will be removed soon. '
      'Please use membership_inference_attack_new.run_attacks'
  )
  metrics = ['auc', 'advantage']

  # Entire data
  result = run_attack(
      loss_train,
      loss_test,
      logits_train,
      logits_test,
      attack_classifiers=attack_classifiers,
      metric=metrics)
  result = utils.prepend_to_keys(result, 'all_')

  # Misclassified examples
  if (labels_train is not None and labels_test is not None and
      logits_train is not None and logits_test is not None):
    result.update(
        run_attack(
            loss_train,
            loss_test,
            logits_train,
            logits_test,
            labels_train,
            labels_test,
            attack_classifiers=attack_classifiers,
            only_misclassified=True,
            metric=metrics))

  # Split per class
  if labels_train is not None and labels_test is not None:
    result.update(
        run_attack(
            loss_train,
            loss_test,
            logits_train,
            logits_test,
            labels_train,
            labels_test,
            by_class=True,
            attack_classifiers=attack_classifiers,
            metric=metrics))

  # Different deciles
  if logits_train is not None and logits_test is not None:
    result.update(
        run_attack(
            loss_train,
            loss_test,
            logits_train,
            logits_test,
            by_percentile=True,
            attack_classifiers=attack_classifiers,
            metric=metrics))

  if decimals is not None:
    result = {k: round(v, decimals) for k, v in result.items()}

  return result


def run_all_attacks_and_create_summary(
    loss_train: np.ndarray = None,
    loss_test: np.ndarray = None,
    logits_train: np.ndarray = None,
    logits_test: np.ndarray = None,
    labels_train: np.ndarray = None,
    labels_test: np.ndarray = None,
    return_dict: bool = True,
    decimals: Union[int, None] = 4) -> Union[Text, Tuple[Text, AnyDict]]:
  """Runs all possible membership inference attack(s) and distill results.

  Check 'run_attack' for detailed information of how attacks are performed
  and evaluated.

  This function internally chooses sane default settings for all attacks and
  returns all possible output combinations.
  For fine grained control and partial attacks, please see `run_attack`.

  Args:
    loss_train: A 1D array containing the individual scalar losses for examples
      used during training.
    loss_test: A 1D array containing the individual scalar losses for examples
      not used during training.
    logits_train: A 2D array (n_train, n_classes) of the individual logits or
      output probabilities of examples used during training.
    logits_test: A 2D array (n_test, n_classes) of the individual logits or
      output probabilities of examples not used during training.
    labels_train: The true labels of the training examples. Labels are only
      needed when `by_class` is specified (i.e., not False).
    labels_test: The true labels of the test examples. Labels are only needed
      when `by_class` is specified (i.e., not False).
    return_dict: Whether to also return a dictionary with the results summarized
      in the summary string.
    decimals: Round all float results to this number of decimals. If decimals is
      None, don't round.

  Returns:
    summarystring: A string with natural language summary of the attacks. In the
      summary string printed numbers will be rounded to `decimal` decimals if
      provided, otherwise will round to 3 diits by default for readability.
    result: a dictionary with all the distilled attack information summarized
      in the summarystring
  """
  print(
      'Deprecation warning: function run_all_attacks_and_create_summary is '
      'deprecated and will be removed soon. '
      'Please use membership_inference_attack_new.run_attacks'
  )
  summary = []
  metrics = ['auc', 'advantage']
  attack_classifiers = ['lr', 'knn']
  results = run_all_attacks(
      loss_train,
      loss_test,
      logits_train,
      logits_test,
      labels_train,
      labels_test,
      attack_classifiers=attack_classifiers,
      decimals=None)
  output = _get_maximum_vulnerability(results, metrics, filterby='all')

  if decimals is not None:
    strdec = decimals
  else:
    strdec = 4

  for metric in metrics:
    summary.append(f'========== {metric.upper()} ==========')
    best_value = output['all-' + metric]['value']
    best_attacker = output['all-' + metric]['attacker']
    summary.append(f'The best attack ({best_attacker}) achieved an {metric} of '
                   f'{best_value:.{strdec}f}.')
    summary.append('')

  classgap = _get_maximum_class_gap_or_none(results, metrics)
  if classgap:
    output.update(classgap)
    for metric in metrics:
      summary.append(f'========== {metric.upper()} per class ==========')
      hi_idx = output[f'max_vuln_class_{metric}']
      lo_idx = output[f'min_vuln_class_{metric}']
      hi = output[f'class_{hi_idx}_{metric}']
      lo = output[f'class_{lo_idx}_{metric}']
      gap = output[f'max_class_gap_{metric}']
      summary.append(f'The most vulnerable class {hi_idx} has {metric} of '
                     f'{hi:.{strdec}f}.')
      summary.append(f'The least vulnerable class {lo_idx} has {metric} of '
                     f'{lo:.{strdec}f}.')
      summary.append(f'=> The maximum gap between class vulnerabilities is '
                     f'{gap:.{strdec}f}.')
      summary.append('')

  misclassified = _get_maximum_vulnerability(
      results, metrics, filterby='misclassified')
  if misclassified:
    for metric in metrics:
      best_value = misclassified['misclassified-' + metric]['value']
      best_attacker = misclassified['misclassified-' + metric]['attacker']
      summary.append(f'========== {metric.upper()} for misclassified '
                     '==========')
      summary.append('Among misclassified examples, the best attack '
                     f'({best_attacker}) achieved an {metric} of '
                     f'{best_value:.{strdec}f}.')
      summary.append('')
    output.update(misclassified)

  n_examples = {k: v for k, v in results.items() if k.endswith('n_examples')}
  if n_examples:
    output.update(n_examples)

  # Flatten remaining dicts in output
  fresh_output = {}
  for k, v in output.items():
    if isinstance(v, dict):
      if k.startswith('all'):
        fresh_output[k[4:]] = v['value']
        fresh_output['best_attacker_' + k[4:]] = v['attacker']
    else:
      fresh_output[k] = v
  output = fresh_output

  if decimals is not None:
    for k, v in output.items():
      if isinstance(v, float):
        output[k] = round(v, decimals)

  summary = '\n'.join(summary)
  if return_dict:
    return summary, output
  else:
    return summary

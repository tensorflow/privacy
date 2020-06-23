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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.utils."""
from absl.testing import absltest

import numpy as np

from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia


def get_result_dict():
  """Get an example result dictionary."""
  return {
      'test_n_examples': np.ones(1),
      'test_examples': np.zeros(1),
      'test_auc': np.ones(1),
      'test_advantage': np.ones(1),
      'all_0-metric': np.array([1]),
      'all_1-metric': np.array([2]),
      'test_2-metric': np.array([3]),
      'test_score': np.array([4]),
  }


def get_test_inputs():
  """Get example inputs for attacks."""
  n_train = n_test = 500
  rng = np.random.RandomState(4)
  loss_train = rng.randn(n_train) - 0.4
  loss_test = rng.randn(n_test) + 0.4
  logits_train = rng.randn(n_train, 5) + 0.2
  logits_test = rng.randn(n_test, 5) - 0.2
  labels_train = np.array([i % 5 for i in range(n_train)])
  labels_test = np.array([(3 * i) % 5 for i in range(n_test)])
  return (loss_train, loss_test, logits_train, logits_test,
          labels_train, labels_test)


class GetVulnerabilityTest(absltest.TestCase):

  def test_get_vulnerabilities(self):
    """Test extraction of vulnerability scores."""
    testdict = get_result_dict()
    for key in ['auc', 'advantage']:
      res = mia._get_vulnerabilities(testdict, key)
      self.assertLen(res, 2)
      self.assertEqual(res[f'test_{key}'], 1)
      self.assertEqual(res['test_n_examples'], 1)

    res = mia._get_vulnerabilities(testdict, ['auc', 'advantage'])
    self.assertLen(res, 3)
    self.assertEqual(res['test_auc'], 1)
    self.assertEqual(res['test_advantage'], 1)
    self.assertEqual(res['test_n_examples'], 1)


class GetMaximumVulnerabilityTest(absltest.TestCase):

  def test_get_maximum_vulnerability(self):
    """Test extraction of maximum vulnerability score."""
    testdict = get_result_dict()
    for i in range(3):
      key = f'{i}-metric'
      res = mia._get_maximum_vulnerability(testdict, key)
      self.assertLen(res, 1)
      self.assertEqual(res[key]['value'], i + 1)
      if i < 2:
        self.assertEqual(res[key]['attacker'], f'all_{i}-metric')
      else:
        self.assertEqual(res[key]['attacker'], 'test_2-metric')

    res = mia._get_maximum_vulnerability(testdict, 'metric')
    self.assertLen(res, 1)
    self.assertEqual(res['metric']['value'], 3)

    res = mia._get_maximum_vulnerability(testdict, ['metric'],
                                         filterby='all')
    self.assertLen(res, 1)
    self.assertEqual(res['all-metric']['value'], 2)

    res = mia._get_maximum_vulnerability(testdict, ['metric', 'score'])
    self.assertLen(res, 2)
    self.assertEqual(res['metric']['value'], 3)
    self.assertEqual(res['score']['value'], 4)
    self.assertEqual(res['score']['attacker'], 'test_score')


class ThresholdAttackLossTest(absltest.TestCase):

  def test_threshold_attack_loss(self):
    """Test simple threshold attack on loss."""
    features = {
        'loss': np.zeros(10),
        'is_train': np.concatenate((np.zeros(5), np.ones(5))),
    }
    res = mia._run_threshold_loss_attack(features)
    for k in res:
      self.assertStartsWith(k, 'thresh_loss')
    self.assertEqual(res['thresh_loss_auc'], 0.5)
    self.assertEqual(res['thresh_loss_advantage'], 0.0)

    rng = np.random.RandomState(4)
    n_train = 1000
    n_test = 500
    loss_train = rng.randn(n_train) - 0.4
    loss_test = rng.randn(n_test) + 0.4
    features = {
        'loss': np.concatenate((loss_train, loss_test)),
        'is_train': np.concatenate((np.ones(n_train), np.zeros(n_test))),
    }
    res = mia._run_threshold_loss_attack(features)
    self.assertBetween(res['thresh_loss_auc'], 0.7, 0.75)
    self.assertBetween(res['thresh_loss_advantage'], 0.3, 0.35)


class ThresholdAttackMaxlogitTest(absltest.TestCase):

  def test_threshold_attack_maxlogits(self):
    """Test simple threshold attack on maximum logit."""
    features = {
        'logits': np.eye(10, 14),
        'is_train': np.concatenate((np.zeros(5), np.ones(5))),
    }
    res = mia._run_threshold_attack_maxlogit(features)
    for k in res:
      self.assertStartsWith(k, 'thresh_maxlogit')
    self.assertEqual(res['thresh_maxlogit_auc'], 0.5)
    self.assertEqual(res['thresh_maxlogit_advantage'], 0.0)

    rng = np.random.RandomState(4)
    n_train = 1000
    n_test = 500
    logits_train = rng.randn(n_train, 12) + 0.2
    logits_test = rng.randn(n_test, 12) - 0.2
    features = {
        'logits': np.concatenate((logits_train, logits_test), axis=0),
        'is_train': np.concatenate((np.ones(n_train), np.zeros(n_test))),
    }
    res = mia._run_threshold_attack_maxlogit(features)
    self.assertBetween(res['thresh_maxlogit_auc'], 0.7, 0.75)
    self.assertBetween(res['thresh_maxlogit_advantage'], 0.3, 0.35)


class TrainedAttackTrivialTest(absltest.TestCase):

  def test_trained_attack(self):
    """Test trained attacks."""
    # Trivially easy problem
    x_train, x_test = np.ones((500, 3)), np.ones((20, 3))
    x_train[:200] *= -1
    x_test[:8] *= -1
    y_train, y_test = np.ones(500).astype(int), np.ones(20).astype(int)
    y_train[:200] = 0
    y_test[:8] = 0
    data = (x_train, y_train), (x_test, y_test)
    for clf in ['lr', 'rf', 'mlp', 'knn']:
      res = mia._run_trained_attack(clf, data, attack_prefix='a-')
      self.assertEqual(res['a-train_auc'], 1)
      self.assertEqual(res['a-test_auc'], 1)
      self.assertEqual(res['a-train_advantage'], 1)
      self.assertEqual(res['a-test_advantage'], 1)


class TrainedAttackRandomFeaturesTest(absltest.TestCase):

  def test_trained_attack(self):
    """Test trained attacks."""
    # Random labels and features
    rng = np.random.RandomState(4)
    x_train, x_test = rng.randn(500, 3), rng.randn(500, 3)
    y_train = rng.binomial(1, 0.5, size=(500,))
    y_test = rng.binomial(1, 0.5, size=(500,))
    data = (x_train, y_train), (x_test, y_test)
    for clf in ['lr', 'rf', 'mlp', 'knn']:
      res = mia._run_trained_attack(clf, data, attack_prefix='a-')
      self.assertBetween(res['a-train_auc'], 0.5, 1.)
      self.assertBetween(res['a-test_auc'], 0.4, 0.6)
      self.assertBetween(res['a-train_advantage'], 0., 1.0)
      self.assertBetween(res['a-test_advantage'], 0., 0.2)


class AttackLossesTest(absltest.TestCase):

  def test_attack(self):
    """Test individual attack function."""
    # losses only, both metrics
    loss_train, loss_test, _, _, _, _ = get_test_inputs()
    res = mia.run_attack(loss_train, loss_test, metric=('auc', 'advantage'))
    self.assertBetween(res['thresh_loss_auc'], 0.7, 0.75)
    self.assertBetween(res['thresh_loss_advantage'], 0.3, 0.35)


class AttackLossesLogitsTest(absltest.TestCase):

  def test_attack(self):
    """Test individual attack function."""
    # losses and logits, two classifiers, single metric
    loss_train, loss_test, logits_train, logits_test, _, _ = get_test_inputs()
    res = mia.run_attack(
        loss_train,
        loss_test,
        logits_train,
        logits_test,
        attack_classifiers=('rf', 'knn'),
        metric='auc')
    self.assertBetween(res['rf_logits_test_auc'], 0.7, 0.9)
    self.assertBetween(res['knn_logits_test_auc'], 0.7, 0.9)
    self.assertBetween(res['rf_logits_loss_test_auc'], 0.7, 0.9)
    self.assertBetween(res['knn_logits_loss_test_auc'], 0.7, 0.9)


class AttackLossesLabelsByClassTest(absltest.TestCase):

  def test_attack(self):
    # losses and labels, single metric, split by class
    loss_train, loss_test, _, _, labels_train, labels_test = get_test_inputs()
    n_train = loss_train.shape[0]
    n_test = loss_test.shape[0]
    res = mia.run_attack(
        loss_train,
        loss_test,
        labels_train=labels_train,
        labels_test=labels_test,
        by_class=True,
        metric='auc')
    self.assertLen(res, 10)
    for k in res:
      self.assertStartsWith(k, 'class_')
      if k.endswith('n_examples'):
        self.assertEqual(int(res[k]), (n_train + n_test) // 5)
      else:
        self.assertBetween(res[k], 0.65, 0.75)


class AttackLossesLabelsSingleClassTest(absltest.TestCase):

  def test_attack(self):
    # losses and labels, both metrics, single class
    loss_train, loss_test, _, _, labels_train, labels_test = get_test_inputs()
    n_train = loss_train.shape[0]
    n_test = loss_test.shape[0]
    res = mia.run_attack(
        loss_train,
        loss_test,
        labels_train=labels_train,
        labels_test=labels_test,
        by_class=2,
        metric=('auc', 'advantage'))
    self.assertLen(res, 3)
    for k in res:
      self.assertStartsWith(k, 'class_2')
      if k.endswith('n_examples'):
        self.assertEqual(int(res[k]), (n_train + n_test) // 5)
      elif k.endswith('advantage'):
        self.assertBetween(res[k], 0.3, 0.5)
      elif k.endswith('auc'):
        self.assertBetween(res[k], 0.7, 0.75)


class AttackLogitsLabelsMisclassifiedTest(absltest.TestCase):

  def test_attack(self):
    # logits and labels, single metric, single classifier, misclassified only
    (_, _, logits_train, logits_test,
     labels_train, labels_test) = get_test_inputs()
    res = mia.run_attack(
        logits_train=logits_train,
        logits_test=logits_test,
        labels_train=labels_train,
        labels_test=labels_test,
        only_misclassified=True,
        attack_classifiers=('lr',),
        metric='advantage')
    self.assertBetween(res['misclassified_lr_logits_test_advantage'], 0.3, 0.8)
    self.assertEqual(res['misclassified_n_examples'], 802)


class AttackLogitsByPrecentileTest(absltest.TestCase):

  def test_attack(self):
    # only logits, single metric, no classifiers, split by deciles
    _, _, logits_train, logits_test, _, _ = get_test_inputs()
    res = mia.run_attack(
        logits_train=logits_train,
        logits_test=logits_test,
        by_percentile=True,
        metric='auc')
    for k in res:
      self.assertStartsWith(k, 'percentile')
      self.assertBetween(res[k], 0.60, 0.75)


if __name__ == '__main__':
  absltest.main()

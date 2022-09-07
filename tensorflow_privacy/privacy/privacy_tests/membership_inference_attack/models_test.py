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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import models
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType


class TrainedAttackerTest(parameterized.TestCase):

  def test_base_attacker_train_and_predict(self):
    base_attacker = models.TrainedAttacker()
    self.assertRaises(NotImplementedError, base_attacker.train_model, [], [])
    self.assertRaises(AssertionError, base_attacker.predict, [])

  def test_predict_before_training(self):
    lr_attacker = models.LogisticRegressionAttacker()
    self.assertRaises(AssertionError, lr_attacker.predict, [])

  def test_create_attacker_data_loss_only(self):
    attack_input = AttackInputData(
        loss_train=np.array([1, 3]), loss_test=np.array([2, 4]))
    attacker_data = models.create_attacker_data(attack_input, 2)
    self.assertLen(attacker_data.features_all, 4)

  def test_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16]))
    attacker_data = models.create_attacker_data(attack_input, balance=False)
    self.assertLen(attacker_data.features_all, 5)
    self.assertLen(attacker_data.fold_indices, 5)
    self.assertEmpty(attacker_data.left_out_indices)

  def test_multilabel_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        labels_train=np.array([[0, 1], [1, 1], [1, 0]]),
        labels_test=np.array([[1, 0], [1, 1]]),
        loss_train=np.array([[1, 3], [6, 7], [8, 9]]),
        loss_test=np.array([[4, 2], [4, 6]]))
    attacker_data = models.create_attacker_data(attack_input, balance=False)
    self.assertLen(attacker_data.features_all, 5)
    self.assertLen(attacker_data.fold_indices, 5)
    self.assertEmpty(attacker_data.left_out_indices)
    self.assertEqual(
        attacker_data.features_all.shape[1],
        attack_input.logits_train.shape[1] + attack_input.loss_train.shape[1])
    self.assertTrue(
        attack_input.is_multilabel_data(),
        msg='Expected multilabel check to pass.')

  def test_unbalanced_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16]))
    attacker_data = models.create_attacker_data(attack_input, balance=True)
    self.assertLen(attacker_data.features_all, 5)
    self.assertLen(attacker_data.fold_indices, 4)
    self.assertLen(attacker_data.left_out_indices, 1)
    self.assertIn(attacker_data.left_out_indices[0], [0, 1, 2])

  def test_balanced_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15], [17, 18]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16, 19]))
    attacker_data = models.create_attacker_data(attack_input)
    self.assertLen(attacker_data.features_all, 6)
    self.assertLen(attacker_data.fold_indices, 6)
    self.assertEmpty(attacker_data.left_out_indices)

  # Parameters for testing: backend.
  @parameterized.named_parameters(
      ('threading_backend', 'threading'),
      ('None_backend', None),
  )
  def test_training_with_backends(self, backend):
    with self.assertLogs(level='INFO') as log:
      attacker = models.create_attacker(
          AttackType.MULTI_LAYERED_PERCEPTRON, backend=backend)
    self.assertIsInstance(attacker, models.MultilayerPerceptronAttacker)
    self.assertLen(log.output, 1)
    self.assertLen(log.records, 1)
    self.assertRegex(log.output[0], r'.+?Using .+? backend for training.')

if __name__ == '__main__':
  absltest.main()

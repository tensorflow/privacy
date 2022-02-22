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

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import utils


class TestLogLoss(parameterized.TestCase):

  @parameterized.named_parameters(
      ('label0', 0,
       np.array([
           4.60517019, 2.30258509, 1.38629436, 0.69314718, 0.28768207,
           0.10536052, 0.01005034
       ])), ('label1', 1,
             np.array([
                 0.01005034, 0.10536052, 0.28768207, 0.69314718, 1.38629436,
                 2.30258509, 4.60517019
             ])))
  def test_log_loss_from_probs_2_classes(self, label, expected_losses):
    pred = np.array([[0.01, 0.99], [0.1, 0.9], [0.25, 0.75], [0.5, 0.5],
                     [0.75, 0.25], [0.9, 0.1], [0.99, 0.01]])
    y = np.full(pred.shape[0], label)
    loss = utils.log_loss(y, pred)
    np.testing.assert_allclose(loss, expected_losses, atol=1e-7)

  @parameterized.named_parameters(
      ('label0', 0, np.array([1.60943791, 0.51082562, 0.51082562, 0.01005034])),
      ('label1', 1, np.array([0.35667494, 1.60943791, 2.30258509, 6.2146081])),
      ('label2', 2, np.array([2.30258509, 1.60943791, 1.2039728, 4.82831374])),
  )
  def test_log_loss_from_probs_3_classes(self, label, expected_losses):
    # Values from http://bit.ly/RJJHWA
    pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3],
                     [0.99, 0.002, 0.008]])
    y = np.full(pred.shape[0], label)
    loss = utils.log_loss(y, pred)
    np.testing.assert_allclose(loss, expected_losses, atol=1e-7)

  @parameterized.named_parameters(
      ('small_value1e-8', 1e-8, 18.42068074),
      ('small_value1e-20', 1e-20, 46.05170186),
      ('small_value1e-50', 1e-50, 115.12925465),
  )
  def test_log_loss_from_probs_boundary(self, small_value, expected_loss):
    pred = np.array([[0., 1]] * 2)
    y = np.array([0, 1])
    loss = utils.log_loss(y, pred, small_value=small_value)
    np.testing.assert_allclose(loss, np.array([expected_loss, 0]), atol=1e-7)

  def test_log_loss_from_logits(self):
    logits = np.array([[1, 2, 0, -1], [1, 2, 0, -1], [-1, 3, 0, 0]])
    labels = np.array([0, 3, 1])
    expected_loss = np.array([1.4401897, 3.4401897, 0.11144278])

    loss = utils.log_loss(labels, logits, from_logits=True)
    np.testing.assert_allclose(expected_loss, loss, atol=1e-7)

  @parameterized.named_parameters(
      ('label0', 0,
       np.array([
           0.2231435513, 1.2039728043, 0.1053605157, 4.6051701860, 0.0020020027,
           0.0080321717
       ])), ('label1', 1,
             np.array([
                 1.6094379124, 0.3566749439, 2.3025850930, 0.0100503359,
                 6.2146080984, 4.8283137373
             ])))
  def test_log_loss_binary_from_probs(self, label, expected_loss):
    pred = np.array([0.2, 0.7, 0.1, 0.99, 0.002, 0.008])
    y = np.full(pred.shape[0], label)
    loss = utils.log_loss(y, pred)
    np.testing.assert_allclose(expected_loss, loss, atol=1e-7)

  @parameterized.named_parameters(
      ('label0', 0, np.array([0.000045398, 0.006715348, 0.6931471825, 5, 10])),
      ('label1', 1, np.array([10, 5, 0.6931471825, 0.006715348, 0.000045398])),
  )
  def test_log_loss_binary_from_logits(self, label, expected_loss):
    pred = np.array([-10, -5, 0., 5, 10])
    y = np.full(pred.shape[0], label)
    loss = utils.log_loss(y, pred, from_logits=True)
    np.testing.assert_allclose(expected_loss, loss, rtol=1e-2)

  @parameterized.named_parameters(
      ('binary_mismatch', np.array([0, 1, 2]), np.ones((3,))),
      ('binary_wrong_label', np.array([-1, 1]), np.ones((2,))),
      ('multiclass_wrong_label', np.array([0, 3]), np.ones((2, 3))),
  )
  def test_log_loss_wrong_classes(self, labels, pred):
    self.assertRaises(ValueError, utils.log_loss, labels=labels, pred=pred)


class TestSquaredLoss(parameterized.TestCase):

  def test_squared_loss(self):
    y_true = np.array([1, 2, 3, 4.])
    y_pred = np.array([4, 3, 2, 1.])
    expected_loss = np.array([9, 1, 1, 9.])
    loss = utils.squared_loss(y_true, y_pred)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-7)


if __name__ == '__main__':
  absltest.main()

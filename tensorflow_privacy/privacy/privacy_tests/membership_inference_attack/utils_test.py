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
    loss = utils.log_loss(y, pred.reshape(-1, 1))
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
    loss = utils.log_loss(y, pred.reshape(-1, 1), from_logits=True)
    np.testing.assert_allclose(expected_loss, loss, rtol=1e-2)

  @parameterized.named_parameters(
      ('binary_mismatch', np.array([0, 1, 2]), np.ones((3,))),
      ('binary_wrong_label', np.array([-1, 1]), np.ones((2,))),
      ('multiclass_wrong_label', np.array([0, 3]), np.ones((2, 3))),
  )
  def test_log_loss_wrong_classes(self, labels, pred):
    self.assertRaises(ValueError, utils.log_loss, labels=labels, pred=pred)

  def test_log_loss_wrong_number_of_example(self):
    labels = np.array([0, 1, 1])
    pred = np.array([0.2])
    self.assertRaises(ValueError, utils.log_loss, labels=labels, pred=pred)


class TestSquaredLoss(parameterized.TestCase):

  def test_squared_loss(self):
    y_true = np.array([1, 2, 3, 4.])
    y_pred = np.array([4, 3, 2, 1.])
    expected_loss = np.array([9, 1, 1, 9.])
    loss = utils.squared_loss(y_true, y_pred)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-7)


class TestMultilabelBCELoss(parameterized.TestCase):

  @parameterized.named_parameters(
      ('probs_example1', np.array(
          [[0, 1, 1], [1, 1, 0]]), np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
       np.array([[0.22314343, 1.20397247, 0.3566748],
                 [0.22314343, 0.51082546, 2.30258409]]), False),
      ('probs_example2', np.array([[0, 1, 0], [1, 1, 0]]),
       np.array([[0.01, 0.02, 0.04], [0.8, 0.7, 0.9]]),
       np.array([[0.01005033, 3.91202251, 0.04082198],
                 [0.22314354, 0.35667493, 2.30258499]]), False),
      ('logits_example1', np.array([[0, 1, 1], [1, 1, 0]]),
       np.array([[-1.2, -0.3, 2.1], [0.0, 0.5, 1.5]]),
       np.array([[0.26328245, 0.85435522, 0.11551951],
                 [0.69314716, 0.47407697, 1.70141322]]), True),
      ('logits_example2', np.array([[0, 1, 0], [1, 1, 0]]),
       np.array([[-1.2, -0.3, 2.1], [0.0, 0.5, 1.5]]),
       np.array([[0.26328245, 0.85435522, 2.21551943],
                 [0.69314716, 0.47407697, 1.70141322]]), True),
  )
  def test_multilabel_bce_loss(self, label, pred, expected_losses, from_logits):
    loss = utils.multilabel_bce_loss(label, pred, from_logits=from_logits)
    np.testing.assert_allclose(loss, expected_losses, atol=1e-6)

  @parameterized.named_parameters(
      ('from_logits_true_and_incorrect_values_example1',
       np.array([[0, 1, 1], [1, 1, 0]
                ]), np.array([[0.2, 0.3, 0.7], [0.8, 0.6, 0.9]]),
       np.array([[0.22314343, 1.20397247, 0.3566748],
                 [0.22314343, 0.51082546, 2.30258409]]), True),
      ('from_logits_true_and_incorrect_values_example2',
       np.array([[0, 1, 0], [1, 1, 0]
                ]), np.array([[0.01, 0.02, 0.04], [0.8, 0.7, 0.9]]),
       np.array([[0.01005033, 3.91202251, 0.04082198],
                 [0.22314354, 0.35667493, 2.30258499]]), True),
  )
  def test_multilabel_bce_loss_incorrect_value(self, label, pred,
                                               expected_losses, from_logits):
    loss = utils.multilabel_bce_loss(label, pred, from_logits=from_logits)
    self.assertFalse(np.allclose(loss, expected_losses))

  @parameterized.named_parameters(
      ('from_logits_false_and_pred_not_in_0to1_example1',
       np.array([[0, 1, 1], [1, 1, 0]
                ]), np.array([[-1.2, -0.3, 2.1], [0.0, 0.5, 1.5]]), False,
       (r'Prediction probabilities are not in \[0, 1\] and '
        '`from_logits` is set to False.')),
      ('labels_not_0_or_1', np.array([[0, 1, 0], [1, 2, 0]]),
       np.array([[-1.2, -0.3, 2.1], [0.0, 0.5, 1.5]]), False,
       ('labels should be in {0, 1}. For multi-label classification the labels '
        'should be multihot encoded.')),
  )
  def test_multilabel_bce_loss_raises(self, label, pred, from_logits, regex):
    self.assertRaisesRegex(ValueError, regex, utils.multilabel_bce_loss, label,
                           pred, from_logits)


if __name__ == '__main__':
  absltest.main()

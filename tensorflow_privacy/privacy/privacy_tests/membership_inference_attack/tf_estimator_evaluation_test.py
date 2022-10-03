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
import numpy as np
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow.compat.v1 import estimator as tf_compat_v1_estimator
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import data_structures
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import tf_estimator_evaluation


class UtilsTest(absltest.TestCase):

  def __init__(self, methodname):
    """Initialize the test class."""
    super().__init__(methodname)

    self.ntrain, self.ntest = 50, 100
    self.nclass = 5
    self.ndim = 10

    # Generate random training and test data
    self.train_data = np.random.rand(self.ntrain, self.ndim)
    self.test_data = np.random.rand(self.ntest, self.ndim)
    self.train_labels = np.random.randint(self.nclass, size=self.ntrain)
    self.test_labels = np.random.randint(self.nclass, size=self.ntest)
    self.sample_weight_train = np.random.rand(self.ntrain)
    self.sample_weight_test = np.random.rand(self.ntest)

    # Define a simple model function
    def model_fn(features, labels, mode):
      """Model function for logistic regression."""
      del labels

      input_layer = tf.reshape(features['x'], [-1, self.ndim])
      logits = tf.keras.layers.Dense(self.nclass)(input_layer)

      # Define the PREDICT mode becasue we only need that
      if mode == tf_estimator.ModeKeys.PREDICT:
        predictions = tf.nn.softmax(logits)
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Define the classifier, input_fn for training and test data
    self.classifier = tf_estimator.Estimator(model_fn=model_fn)
    self.input_fn_train = tf_compat_v1_estimator.inputs.numpy_input_fn(
        x={'x': self.train_data},
        y=self.train_labels,
        num_epochs=1,
        shuffle=False)
    self.input_fn_test = tf_compat_v1_estimator.inputs.numpy_input_fn(
        x={'x': self.test_data},
        y=self.test_labels,
        num_epochs=1,
        shuffle=False)

  def test_calculate_losses(self):
    """Test calculating the loss."""
    pred, loss = tf_estimator_evaluation.calculate_losses(
        self.classifier, self.input_fn_train, self.train_labels)
    self.assertEqual(pred.shape, (self.ntrain, self.nclass))
    self.assertEqual(loss.shape, (self.ntrain,))

    pred, loss = tf_estimator_evaluation.calculate_losses(
        self.classifier,
        self.input_fn_train,
        self.train_labels,
        sample_weight=self.sample_weight_train)
    self.assertEqual(pred.shape, (self.ntrain, self.nclass))
    self.assertEqual(loss.shape, (self.ntrain,))

    pred, loss = tf_estimator_evaluation.calculate_losses(
        self.classifier, self.input_fn_test, self.test_labels)
    self.assertEqual(pred.shape, (self.ntest, self.nclass))
    self.assertEqual(loss.shape, (self.ntest,))

  def test_run_attack_helper(self):
    """Test the attack."""
    results = tf_estimator_evaluation.run_attack_helper(
        self.classifier,
        self.input_fn_train,
        self.input_fn_test,
        self.train_labels,
        self.test_labels,
        self.sample_weight_train,
        self.sample_weight_test,
        attack_types=[data_structures.AttackType.THRESHOLD_ATTACK])
    self.assertIsInstance(results, data_structures.AttackResults)
    att_types, att_slices, att_metrics, att_values = data_structures.get_flattened_attack_metrics(
        results)
    self.assertLen(att_types, 2)
    self.assertLen(att_slices, 2)
    self.assertLen(att_metrics, 2)
    self.assertLen(att_values, 3)  # Attacker Advantage, AUC, PPV

  def test_run_attack_helper_with_sample_weights(self):
    """Test the attack."""
    results = tf_estimator_evaluation.run_attack_helper(
        self.classifier,
        self.input_fn_train,
        self.input_fn_test,
        self.train_labels,
        self.test_labels,
        in_train_sample_weight=self.sample_weight_train,
        out_train_sample_weight=self.sample_weight_test,
        attack_types=[data_structures.AttackType.THRESHOLD_ATTACK])
    self.assertIsInstance(results, data_structures.AttackResults)
    att_types, att_slices, att_metrics, att_values = data_structures.get_flattened_attack_metrics(
        results)
    self.assertLen(att_types, 2)
    self.assertLen(att_slices, 2)
    self.assertLen(att_metrics, 2)
    self.assertLen(att_values, 3)  # Attacker Advantage, AUC, PPV

  def test_run_attack_on_tf_estimator_model(self):
    """Test the attack on the final models."""

    def input_fn_constructor(x, y):
      return tf_compat_v1_estimator.inputs.numpy_input_fn(
          x={'x': x}, y=y, shuffle=False)

    results = tf_estimator_evaluation.run_attack_on_tf_estimator_model(
        self.classifier, (self.train_data, self.train_labels),
        (self.test_data, self.test_labels),
        input_fn_constructor,
        attack_types=[data_structures.AttackType.THRESHOLD_ATTACK])
    self.assertIsInstance(results, data_structures.AttackResults)
    att_types, att_slices, att_metrics, att_values = data_structures.get_flattened_attack_metrics(
        results)
    self.assertLen(att_types, 2)
    self.assertLen(att_slices, 2)
    self.assertLen(att_metrics, 2)
    self.assertLen(att_values, 3)  # Attacker Advantage, AUC, PPV

  def test_run_attack_on_tf_estimator_model_with_sample_weights(self):
    """Test the attack on the final models."""

    def input_fn_constructor(x, y):
      return tf_compat_v1_estimator.inputs.numpy_input_fn(
          x={'x': x}, y=y, shuffle=False)

    results = tf_estimator_evaluation.run_attack_on_tf_estimator_model(
        self.classifier,
        (self.train_data, self.train_labels, self.sample_weight_train),
        (self.test_data, self.test_labels),
        input_fn_constructor,
        attack_types=[data_structures.AttackType.THRESHOLD_ATTACK])
    self.assertIsInstance(results, data_structures.AttackResults)
    att_types, att_slices, att_metrics, att_values = data_structures.get_flattened_attack_metrics(
        results)
    self.assertLen(att_types, 2)
    self.assertLen(att_slices, 2)
    self.assertLen(att_metrics, 2)
    self.assertLen(att_values, 3)  # Attacker Advantage, AUC, PPV


if __name__ == '__main__':
  absltest.main()

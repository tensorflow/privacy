# Copyright 2023, The TensorFlow Authors.
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
"""Tests for LinearClassifier."""

import functools

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.estimators import test_utils
from tensorflow_privacy.privacy.estimators.v1 import linear
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

# pylint: disable=g-deprecated-tf-checker


class DPLinearClassifierClassifierTest(
    tf.test.TestCase, parameterized.TestCase
):
  """Tests for DP-enabled LinearClassifier."""

  @parameterized.named_parameters(
      ('BinaryClassLinear 1', 2, 1),
      ('BinaryClassLinear 4', 2, 4),
      ('MultiClassLinear 3', 3, 1),
      ('MultiClassLinear 4', 4, 1),
      ('MultiClassLinear 4 1', 4, 2),
  )
  def testRunsWithoutErrors(self, n_classes, num_microbatches):
    train_features, train_labels = test_utils.make_input_data(256, n_classes)
    feature_columns = []
    for key in train_features:
      feature_columns.append(tf.feature_column.numeric_column(key=key))  # pylint: disable=g-deprecated-tf-checker

    optimizer = functools.partial(
        DPGradientDescentGaussianOptimizer,
        learning_rate=0.5,
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
    )

    classifier = linear.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=n_classes,
        optimizer=optimizer,
        loss_reduction=tf.compat.v1.losses.Reduction.SUM,
    )

    classifier.train(
        input_fn=test_utils.make_input_fn(
            train_features, train_labels, True, 16
        )
    )

    test_features, test_labels = test_utils.make_input_data(64, n_classes)
    classifier.evaluate(
        input_fn=test_utils.make_input_fn(test_features, test_labels, False, 16)
    )

    predict_features, predict_labels = test_utils.make_input_data(64, n_classes)
    classifier.predict(
        input_fn=test_utils.make_input_fn(
            predict_features, predict_labels, False
        )
    )


if __name__ == '__main__':
  tf.test.main()

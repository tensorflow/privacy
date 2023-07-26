# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DP version of LinearClassifier v1."""

import tensorflow as tf
from tensorflow_privacy.privacy.estimators.v1 import head as head_lib
from tensorflow_estimator.python.estimator import estimator  # pylint: disable=g-deprecated-tf-checker
from tensorflow_estimator.python.estimator.canned import linear  # pylint: disable=g-deprecated-tf-checker


class LinearClassifier(estimator.Estimator):
  """DP version of `tf.compat.v1.estimator.LinearClassifier`."""

  def __init__(
      self,
      feature_columns,
      model_dir=None,
      n_classes=2,
      weight_column=None,
      label_vocabulary=None,
      optimizer='Ftrl',
      config=None,
      partitioner=None,
      warm_start_from=None,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM,  # For scalar summary.
      sparse_combiner='sum',
  ):
    """See `tf.compat.v1.estimator.LinearClassifier`."""
    linear._validate_linear_sdca_optimizer_for_linear_classifier(  # pylint: disable=protected-access
        feature_columns=feature_columns,
        n_classes=n_classes,
        optimizer=optimizer,
        sparse_combiner=sparse_combiner,
    )
    estimator._canned_estimator_api_gauge.get_cell('Classifier').set('Linear')  # pylint: disable=protected-access

    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes, weight_column, label_vocabulary, loss_reduction
    )

    def _model_fn(features, labels, mode, config):
      """Call the defined shared _linear_model_fn."""
      return linear._linear_model_fn(  # pylint: disable=protected-access
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          partitioner=partitioner,
          config=config,
          sparse_combiner=sparse_combiner,
      )

    super(LinearClassifier, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from,
    )

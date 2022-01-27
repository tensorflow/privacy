# Copyright 2020, The TensorFlow Privacy Authors.
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
"""TensorFlow Privacy library v1 imports.

This module includes classes designed to be compatible with TF1, based on
`tf.compat.v1.train.Optimizer` and `tf.estimator.Estimator`.
"""

import sys

# pylint: disable=g-import-not-at-top

if hasattr(sys, 'skip_tf_privacy_import'):  # Useful for standalone scripts.
  pass
else:
  # Estimators
  from tensorflow_privacy.privacy.estimators.v1.dnn import DNNClassifier as DNNClassifierV1

  # Optimizers
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdagradGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdagradOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import make_optimizer_class

  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdagradOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdamOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPSGDOptimizer

  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdagrad
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPSGD
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import make_vectorized_optimizer_class

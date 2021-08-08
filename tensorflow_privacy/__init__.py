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
"""TensorFlow Privacy library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow_privacy.version import __version__  # pylint: disable=g-bad-import-order

# pylint: disable=g-import-not-at-top

if hasattr(sys, 'skip_tf_privacy_import'):  # Useful for standalone scripts.
  pass
else:
  # TensorFlow v1 imports
  import tensorflow_privacy.v1

  # Analysis
  from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
  from tensorflow_privacy.privacy.analysis.privacy_ledger import GaussianSumQueryEntry
  from tensorflow_privacy.privacy.analysis.privacy_ledger import PrivacyLedger
  from tensorflow_privacy.privacy.analysis.privacy_ledger import QueryWithLedger
  from tensorflow_privacy.privacy.analysis.privacy_ledger import SampleEntry
  from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_heterogenous_rdp
  from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
  from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp_from_ledger
  from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

  # DPQuery classes
  from tensorflow_privacy.privacy.dp_query.dp_query import DPQuery
  from tensorflow_privacy.privacy.dp_query.dp_query import SumAggregationDPQuery
  from tensorflow_privacy.privacy.dp_query.discrete_gaussian_query import DiscreteGaussianSumQuery
  from tensorflow_privacy.privacy.dp_query.distributed_discrete_gaussian_query import DistributedDiscreteGaussianSumQuery
  from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianSumQuery
  from tensorflow_privacy.privacy.dp_query.nested_query import NestedQuery
  from tensorflow_privacy.privacy.dp_query.no_privacy_query import NoPrivacyAverageQuery
  from tensorflow_privacy.privacy.dp_query.no_privacy_query import NoPrivacySumQuery
  from tensorflow_privacy.privacy.dp_query.normalized_query import NormalizedQuery
  from tensorflow_privacy.privacy.dp_query.quantile_estimator_query import QuantileEstimatorQuery
  from tensorflow_privacy.privacy.dp_query.quantile_estimator_query import NoPrivacyQuantileEstimatorQuery
  from tensorflow_privacy.privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipSumQuery
  from tensorflow_privacy.privacy.dp_query import tree_aggregation
  from tensorflow_privacy.privacy.dp_query.tree_aggregation_query import TreeCumulativeSumQuery
  from tensorflow_privacy.privacy.dp_query.tree_aggregation_query import TreeResidualSumQuery

  # Estimators
  from tensorflow_privacy.privacy.estimators.dnn import DNNClassifier

  # Keras Models
  from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPModel
  from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential
  from tensorflow_privacy.privacy.keras_models.dp_keras_model import make_dp_model_class

  # Optimizers
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdagradOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class

  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdagradOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import make_vectorized_keras_optimizer_class

  try:
    from tensorflow_privacy.privacy.bolt_on.models import BoltOnModel
    from tensorflow_privacy.privacy.bolt_on.optimizers import BoltOn
    from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin
    from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexBinaryCrossentropy
    from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexHuber
  except ImportError:
    # module `bolt_on` not yet available in this version of TF Privacy
    pass

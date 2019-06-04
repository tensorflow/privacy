# Copyright 2019, The TensorFlow Privacy Authors.
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

import sys

# pylint: disable=g-import-not-at-top

if hasattr(sys, 'skip_tf_privacy_import'):  # Useful for standalone scripts.
  pass
else:
  from privacy.analysis.privacy_ledger import GaussianSumQueryEntry
  from privacy.analysis.privacy_ledger import PrivacyLedger
  from privacy.analysis.privacy_ledger import QueryWithLedger
  from privacy.analysis.privacy_ledger import SampleEntry

  from privacy.dp_query.dp_query import DPQuery
  from privacy.dp_query.gaussian_query import GaussianAverageQuery
  from privacy.dp_query.gaussian_query import GaussianSumQuery
  from privacy.dp_query.nested_query import NestedQuery
  from privacy.dp_query.no_privacy_query import NoPrivacyAverageQuery
  from privacy.dp_query.no_privacy_query import NoPrivacySumQuery
  from privacy.dp_query.normalized_query import NormalizedQuery
  from privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipSumQuery
  from privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipAverageQuery

  from privacy.optimizers.dp_optimizer import DPAdagradGaussianOptimizer
  from privacy.optimizers.dp_optimizer import DPAdagradOptimizer
  from privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
  from privacy.optimizers.dp_optimizer import DPAdamOptimizer
  from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
  from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer

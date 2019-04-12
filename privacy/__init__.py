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
  from tensorflow_privacy.privacy.analysis.privacy_ledger import DummyLedger
  from tensorflow_privacy.privacy.analysis.privacy_ledger import GaussianSumQueryEntry
  from tensorflow_privacy.privacy.analysis.privacy_ledger import PrivacyLedger
  from tensorflow_privacy.privacy.analysis.privacy_ledger import QueryWithLedger
  from tensorflow_privacy.privacy.analysis.privacy_ledger import SampleEntry

  from tensorflow_privacy.privacy.dp_query.dp_query import DPQuery
  from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianAverageQuery
  from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianSumQuery
  from tensorflow_privacy.privacy.dp_query.nested_query import NestedQuery
  from tensorflow_privacy.privacy.dp_query.no_privacy_query import NoPrivacyAverageQuery
  from tensorflow_privacy.privacy.dp_query.no_privacy_query import NoPrivacySumQuery
  from tensorflow_privacy.privacy.dp_query.normalized_query import NormalizedQuery

  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdagradGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdagradOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
  from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer

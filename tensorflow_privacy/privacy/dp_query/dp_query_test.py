# Copyright 2022, The TensorFlow Authors.
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

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import no_privacy_query


class SumAggregationQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_initial_sample_state_works_on_tensorspecs(self):
    query = no_privacy_query.NoPrivacySumQuery()
    template = tf.TensorSpec.from_tensor(tf.constant([1.0, 2.0]))
    sample_state = query.initial_sample_state(template)
    expected = [0.0, 0.0]
    self.assertAllClose(sample_state, expected)


if __name__ == '__main__':
  tf.test.main()

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

import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_test_utils as ctu
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import layer_normalization_test


class GradNormTpuTest(layer_normalization_test.GradNormTest):

  def setUp(self):
    super().setUp()
    self.strategy = ctu.create_tpu_strategy()
    self.assertIn('TPU', self.strategy.extended.worker_devices[0])


if __name__ == '__main__':
  tf.test.main()

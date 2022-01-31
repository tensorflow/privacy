# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib


class ComputeNoiseFromBudgetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Test0', 60000, 150, 0.941870567, 15, 1e-5, 1e-5, 1.3),
      ('Test1', 100000, 100, 1.70928734, 30, 1e-7, 1e-6, 1.0),
      ('Test2', 100000000, 1024, 5907984.81339406, 10, 1e-7, 1e-5, 0.1),
      ('Test3', 100000000, 1024, 5907984.81339406, 10, 1e-7, 1, 0),
  )
  def test_compute_noise(self, n, batch_size, target_epsilon, epochs, delta,
                         min_noise, expected_noise):
    self.skipTest('Disable test.')
    target_noise = compute_noise_from_budget_lib.compute_noise(
        n, batch_size, target_epsilon, epochs, delta, min_noise)
    self.assertAlmostEqual(target_noise, expected_noise)


if __name__ == '__main__':
  absltest.main()

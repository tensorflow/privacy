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

import math

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


class ComputeDpSgdPrivacyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Test0', 60000, 150, 1.3, 15, 1e-5, 0.7242234026109595, 19.0),
      ('Test1', 100000, 100, 1.0, 30, 1e-7, 1.4154988495444845, 13.0),
      ('Test2', 100000000, 1024, 0.1, 10, 1e-7, 5907982.31138195, 1.25),
  )
  def test_compute_dp_sgd_privacy(self, n, batch_size, noise_multiplier, epochs,
                                  delta, expected_eps, expected_order):
    eps, order = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n, batch_size, noise_multiplier, epochs, delta)
    self.assertAlmostEqual(eps, expected_eps)
    self.assertEqual(order, expected_order)

    # We perform an additional sanity check on the hard-coded test values.
    # We do a back-of-the-envelope calculation to obtain a lower bound.
    # Specifically, we make the approximation that subsampling a q-fraction is
    # equivalent to multiplying noise scale by 1/q.
    # This is only an approximation, but can be justified by the central limit
    # theorem in the Gaussian Differential Privacy framework; see
    # https://arxiv.org/1911.11607
    # The approximation error is one-sided and provides a lower bound, which is
    # the basis of this sanity check. This is confirmed in the above paper.
    q = batch_size / n
    steps = epochs * n / batch_size
    sigma = noise_multiplier * math.sqrt(steps) / q
    # We compute the optimal guarantee for Gaussian
    # using https://arxiv.org/abs/1805.06530 Theorem 8 (in v2).
    low_delta = .5 * math.erfc((eps * sigma - .5 / sigma) / math.sqrt(2))
    if eps < 100:  # Skip this if it causes overflow; error is minor.
      low_delta -= math.exp(eps) * .5 * math.erfc(
          (eps * sigma + .5 / sigma) / math.sqrt(2))
    self.assertLessEqual(low_delta, delta)


if __name__ == '__main__':
  absltest.main()

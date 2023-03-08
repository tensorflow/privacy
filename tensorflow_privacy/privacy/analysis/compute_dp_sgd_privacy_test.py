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


_example_privacy = compute_dp_sgd_privacy_lib._compute_dp_sgd_example_privacy
_user_privacy = compute_dp_sgd_privacy_lib._compute_dp_sgd_user_privacy


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

  @parameterized.named_parameters(
      ('num_epochs_negative', dict(num_epochs=-1.0)),
      ('noise_multiplier_negative', dict(noise_multiplier=-1.0)),
      ('example_delta_negative', dict(example_delta=-0.5)),
      ('example_delta_excessive', dict(example_delta=1.5)),
  )
  def test_compute_dp_sgd_example_privacy_bad_args(self, override_args):
    args = dict(num_epochs=1.0, noise_multiplier=1.0, example_delta=1.0)
    args.update(override_args)
    with self.assertRaises(ValueError):
      _example_privacy(**args)

  @parameterized.named_parameters(
      ('no_microbatching_no_subsampling', False, None, 10.8602036),
      ('microbatching_no_subsampling', True, None, 26.2880374),
      ('no_microbatching_with_subsampling', False, 1e-2, 3.2391922),
      ('microbatching_with_subsampling', True, 1e-2, 22.5970358),
  )
  def test_compute_dp_sgd_example_privacy(
      self, used_microbatching, poisson_subsampling_probability, expected_eps
  ):
    num_epochs = 1.2
    noise_multiplier = 0.7
    example_delta = 1e-5
    eps = _example_privacy(
        num_epochs,
        noise_multiplier,
        example_delta,
        used_microbatching,
        poisson_subsampling_probability,
    )
    self.assertAlmostEqual(eps, expected_eps)

  @parameterized.named_parameters(
      ('num_epochs_negative', dict(num_epochs=-1.0)),
      ('noise_multiplier_negative', dict(noise_multiplier=-1.0)),
      ('example_delta_negative', dict(user_delta=-0.5)),
      ('example_delta_excessive', dict(user_delta=1.5)),
      ('max_examples_per_user_negative', dict(max_examples_per_user=-1)),
  )
  def test_compute_dp_sgd_user_privacy_bad_args(self, override_args):
    args = dict(
        num_epochs=1.0,
        noise_multiplier=1.0,
        user_delta=1.0,
        max_examples_per_user=3,
    )
    args.update(override_args)
    with self.assertRaises(ValueError):
      _user_privacy(**args)

  def test_user_privacy_one_example_per_user(self):
    num_epochs = 1.2
    noise_multiplier = 0.7
    delta = 1e-5

    example_eps = _example_privacy(num_epochs, noise_multiplier, delta)
    user_eps = _user_privacy(
        num_epochs,
        noise_multiplier,
        delta,
        max_examples_per_user=1,
    )
    self.assertEqual(user_eps, example_eps)

  @parameterized.parameters((0.9, 2), (1.1, 3), (2.3, 13))
  def test_user_privacy_epsilon_delta_consistency(self, z, k):
    """Tests example/user epsilons consistent with Vadhan (2017) Lemma 2.2."""
    num_epochs = 5
    user_delta = 1e-6
    q = 2e-4
    user_eps = _user_privacy(
        num_epochs,
        noise_multiplier=z,
        user_delta=user_delta,
        max_examples_per_user=k,
        poisson_subsampling_probability=q,
    )
    example_eps = _example_privacy(
        num_epochs,
        noise_multiplier=z,
        example_delta=user_delta / (k * math.exp(user_eps)),
        poisson_subsampling_probability=q,
    )
    self.assertAlmostEqual(user_eps, example_eps * k)


if __name__ == '__main__':
  absltest.main()

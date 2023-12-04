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
_RDP = compute_dp_sgd_privacy_lib.AccountantType.RDP
_PLD = compute_dp_sgd_privacy_lib.AccountantType.PLD


DP_SGD_STATEMENT_KWARGS = dict(
    number_of_examples=10000,
    batch_size=64,
    num_epochs=5.0,
    noise_multiplier=2.0,
    delta=1e-6,
)


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
      ('no_microbatching_no_subsampling_rdp', False, None, _RDP, 10.8602036),
      ('microbatching_no_subsampling_rdp', True, None, _RDP, 26.2880374),
      ('no_microbatching_with_subsampling_rdp', False, 1e-2, _RDP, 3.2391922),
      ('microbatching_with_subsampling_rdp', True, 1e-2, _RDP, 22.5970358),
      ('no_microbatching_no_subsampling_pld', False, None, _PLD, 10.1224946),
      ('microbatching_no_subsampling_pld', True, None, _PLD, 24.7160779),
      ('no_microbatching_with_subsampling_pld', False, 1e-2, _PLD, 2.4612381),
      ('microbatching_with_subsampling_pld', True, 1e-2, _PLD, 18.6977407),
  )
  def test_compute_dp_sgd_example_privacy(
      self,
      used_microbatching,
      poisson_subsampling_probability,
      accountant_type,
      expected_eps,
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
        accountant_type,
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

  @parameterized.named_parameters(('RDP', _RDP), ('PLD', _PLD))
  def test_user_privacy_one_example_per_user(self, accountant_type):
    num_epochs = 1.2
    noise_multiplier = 0.7
    delta = 1e-5

    example_eps = _example_privacy(
        num_epochs, noise_multiplier, delta, accountant_type=accountant_type
    )
    user_eps = _user_privacy(
        num_epochs,
        noise_multiplier,
        delta,
        max_examples_per_user=1,
        accountant_type=accountant_type,
    )
    self.assertEqual(user_eps, example_eps)

  @parameterized.parameters((0.9, 2), (1.1, 3), (2.3, 13))
  def test_user_privacy_epsilon_delta_consistency(
      self, noise_multiplier, max_examples_per_user
  ):
    """Tests example/user epsilons consistent with Vadhan (2017) Lemma 2.2."""
    num_epochs = 5
    example_delta = 1e-8
    q = 2e-4
    example_eps = _example_privacy(
        num_epochs,
        noise_multiplier=noise_multiplier,
        example_delta=example_delta,
        poisson_subsampling_probability=q,
        accountant_type=_RDP,
    )

    user_delta = math.exp(
        math.log(example_delta)
        + compute_dp_sgd_privacy_lib._logexpm1(
            max_examples_per_user * example_eps
        )
        - compute_dp_sgd_privacy_lib._logexpm1(example_eps)
    )
    user_eps = _user_privacy(
        num_epochs,
        noise_multiplier=noise_multiplier,
        user_delta=user_delta,
        max_examples_per_user=max_examples_per_user,
        poisson_subsampling_probability=q,
        accountant_type=_RDP,
    )
    self.assertAlmostEqual(user_eps, example_eps * max_examples_per_user)

  def test_dp_sgd_privacy_statement_no_user_dp_with_rdp(self):
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        **DP_SGD_STATEMENT_KWARGS,
        accountant_type=_RDP,
    )
    expected_statement = """\
DP-SGD performed over 10000 examples with 64 examples per iteration, noise
multiplier 2.0 for 5.0 epochs with microbatching, and no bound on number of
examples per user.

This privacy guarantee protects the release of all model checkpoints in addition
to the final model.

Example-level DP with add-or-remove-one adjacency at delta = 1e-06 computed with
RDP accounting:
    Epsilon with each example occurring once per epoch:        13.376
    Epsilon assuming Poisson sampling (*):                      1.616

No user-level privacy guarantee is possible without a bound on the number of
examples per user.

(*) Poisson sampling is not usually done in training pipelines, but assuming
that the data was randomly shuffled, it is believed that the actual epsilon
should be closer to this value than the conservative assumption of an arbitrary
data order.
"""
    self.assertEqual(statement, expected_statement)

  def test_dp_sgd_privacy_statement_user_dp_with_rdp(self):
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        **DP_SGD_STATEMENT_KWARGS,
        max_examples_per_user=3,
        accountant_type=_RDP,
    )
    expected_statement = """\
DP-SGD performed over 10000 examples with 64 examples per iteration, noise
multiplier 2.0 for 5.0 epochs with microbatching, and at most 3 examples per
user.

This privacy guarantee protects the release of all model checkpoints in addition
to the final model.

Example-level DP with add-or-remove-one adjacency at delta = 1e-06 computed with
RDP accounting:
    Epsilon with each example occurring once per epoch:        13.376
    Epsilon assuming Poisson sampling (*):                      1.616

User-level DP with add-or-remove-one adjacency at delta = 1e-06 computed using
RDP accounting and group privacy:
    Epsilon with each example occurring once per epoch:        85.940
    Epsilon assuming Poisson sampling (*):                      6.425

(*) Poisson sampling is not usually done in training pipelines, but assuming
that the data was randomly shuffled, it is believed that the actual epsilon
should be closer to this value than the conservative assumption of an arbitrary
data order.
"""
    self.assertEqual(statement, expected_statement)

  def test_dp_sgd_privacy_statement_user_dp_infinite_with_rdp(self):
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        **DP_SGD_STATEMENT_KWARGS,
        max_examples_per_user=10,
        accountant_type=_RDP,
    )
    expected_statement = """\
DP-SGD performed over 10000 examples with 64 examples per iteration, noise
multiplier 2.0 for 5.0 epochs with microbatching, and at most 10 examples per
user.

This privacy guarantee protects the release of all model checkpoints in addition
to the final model.

Example-level DP with add-or-remove-one adjacency at delta = 1e-06 computed with
RDP accounting:
    Epsilon with each example occurring once per epoch:        13.376
    Epsilon assuming Poisson sampling (*):                      1.616

User-level DP with add-or-remove-one adjacency at delta = 1e-06 computed using
RDP accounting and group privacy:
    Epsilon with each example occurring once per epoch:      inf (**)
    Epsilon assuming Poisson sampling (*):                   inf (**)

(*) Poisson sampling is not usually done in training pipelines, but assuming
that the data was randomly shuffled, it is believed that the actual epsilon
should be closer to this value than the conservative assumption of an arbitrary
data order.

(**) A finite example-level epsilon implies a finite user-level epsilon at any
`max_examples_per_user`, but because conversion from example-level to user-level
DP is not exact, it is possible for the upper bound on the user-level epsilon to
still be infinite.
"""
    self.assertEqual(statement, expected_statement)

  def test_dp_sgd_privacy_statement_no_user_dp_with_pld(self):
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        **DP_SGD_STATEMENT_KWARGS,
        accountant_type=_PLD,
    )
    expected_statement = """\
DP-SGD performed over 10000 examples with 64 examples per iteration, noise
multiplier 2.0 for 5.0 epochs with microbatching, and no bound on number of
examples per user.

This privacy guarantee protects the release of all model checkpoints in addition
to the final model.

Example-level DP with add-or-remove-one adjacency at delta = 1e-06 computed with
PLD accounting:
    Epsilon with each example occurring once per epoch:        12.595
    Epsilon assuming Poisson sampling (*):                      1.199

No user-level privacy guarantee is possible without a bound on the number of
examples per user.

(*) Poisson sampling is not usually done in training pipelines, but assuming
that the data was randomly shuffled, it is believed that the actual epsilon
should be closer to this value than the conservative assumption of an arbitrary
data order.
"""
    self.assertEqual(statement, expected_statement)

  def test_dp_sgd_privacy_statement_user_dp_with_pld(self):
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        **DP_SGD_STATEMENT_KWARGS,
        max_examples_per_user=3,
        accountant_type=_PLD,
    )
    expected_statement = """\
DP-SGD performed over 10000 examples with 64 examples per iteration, noise
multiplier 2.0 for 5.0 epochs with microbatching, and at most 3 examples per
user.

This privacy guarantee protects the release of all model checkpoints in addition
to the final model.

Example-level DP with add-or-remove-one adjacency at delta = 1e-06 computed with
PLD accounting:
    Epsilon with each example occurring once per epoch:        12.595
    Epsilon assuming Poisson sampling (*):                      1.199

User-level DP epsilon computation is not supported for PLD accounting at this
time. Use RDP accounting to obtain user-level DP guarantees.

(*) Poisson sampling is not usually done in training pipelines, but assuming
that the data was randomly shuffled, it is believed that the actual epsilon
should be closer to this value than the conservative assumption of an arbitrary
data order.
"""
    self.assertEqual(statement, expected_statement)


if __name__ == '__main__':
  absltest.main()

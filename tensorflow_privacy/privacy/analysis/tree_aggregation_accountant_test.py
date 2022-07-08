# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized
import dp_accounting
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import tree_aggregation_accountant


class TreeAggregationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('eps20', 1.13, 19.74), ('eps2', 8.83, 2.04))
  def test_compute_eps_tree(self, noise_multiplier, eps):
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    # This tests is based on the StackOverflow setting in "Practical and
    # Private (Deep) Learning without Sampling or Shuffling". The calculated
    # epsilon could be better as the method in this package keeps improving.
    steps_list, target_delta = 1600, 1e-6
    rdp = tree_aggregation_accountant.compute_rdp_tree_restart(
        noise_multiplier, steps_list, orders)
    new_eps = dp_accounting.rdp.compute_epsilon(orders, rdp, target_delta)[0]
    self.assertLess(new_eps, eps)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compose_tree_rdp(self, steps_list):
    noise_multiplier, orders = 0.1, 1
    rdp_list = [
        tree_aggregation_accountant.compute_rdp_tree_restart(
            noise_multiplier, steps, orders) for steps in steps_list
    ]
    rdp_composed = tree_aggregation_accountant.compute_rdp_tree_restart(
        noise_multiplier, steps_list, orders)
    self.assertAllClose(rdp_composed, sum(rdp_list), rtol=1e-12)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compute_eps_tree_decreasing(self, steps_list):
    # Test privacy epsilon decreases with noise multiplier increasing when
    # keeping other parameters the same.
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    target_delta = 1e-6
    prev_eps = tree_aggregation_accountant.compute_rdp_tree_restart(
        0, steps_list, orders)
    for noise_multiplier in [0.1 * x for x in range(1, 100, 5)]:
      rdp = tree_aggregation_accountant.compute_rdp_tree_restart(
          noise_multiplier, steps_list, orders)
      eps = dp_accounting.rdp.compute_epsilon(orders, rdp, target_delta)[0]
      self.assertLess(eps, prev_eps)
      prev_eps = eps

  @parameterized.named_parameters(
      ('negative_noise', -1, 3, 1),
      ('empty_steps', 1, [], 1),
      ('negative_steps', 1, -3, 1),
  )
  def test_compute_rdp_tree_restart_raise(self, noise_multiplier, steps_list,
                                          orders):
    with self.assertRaisesRegex(ValueError, 'must be'):
      tree_aggregation_accountant.compute_rdp_tree_restart(
          noise_multiplier, steps_list, orders)

  @parameterized.named_parameters(
      ('t100n0.1', 100, 0.1),
      ('t1000n0.01', 1000, 0.01),
  )
  def test_no_tree_no_sampling(self, total_steps, noise_multiplier):
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    tree_rdp = tree_aggregation_accountant.compute_rdp_tree_restart(
        noise_multiplier, [1] * total_steps, orders)
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(
        dp_accounting.GaussianDpEvent(noise_multiplier), total_steps)
    rdp = accountant._rdp  # pylint: disable=protected-access
    self.assertAllClose(tree_rdp, rdp, rtol=1e-12)

  @parameterized.named_parameters(
      ('negative_noise', -1, 3, 1, 1),
      ('negative_steps', 0.1, -3, 1, 1),
      ('negative_part', 0.1, 3, -1, 1),
      ('negative_sep', 0.1, 3, 1, -1),
  )
  def test_compute_rdp_single_tree_raise(self, noise_multiplier, total_steps,
                                         max_participation, min_separation):
    orders = 1
    with self.assertRaisesRegex(ValueError, 'must be'):
      tree_aggregation_accountant.compute_rdp_single_tree(
          noise_multiplier, total_steps, max_participation, min_separation,
          orders)

  @parameterized.named_parameters(
      ('3', 3),
      ('8', 8),
      ('11', 11),
      ('19', 19),
  )
  def test_max_tree_sensitivity_square_sum_every_step(self, steps):
    max_participation, min_separation = steps, 0
    # If a sample will appear in every leaf node, we can infer the total
    # sensitivity by adding all the nodes.
    steps_bin = bin(steps)[2:]
    depth = [
        len(steps_bin) - 1 - i for i, v in enumerate(steps_bin) if v == '1'
    ]
    expected = sum([2**d * (2**(d + 1) - 1) for d in depth])
    self.assertEqual(
        expected,
        tree_aggregation_accountant._max_tree_sensitivity_square_sum(
            max_participation, min_separation, steps))

  @parameterized.named_parameters(
      ('11', 11),
      ('19', 19),
      ('200', 200),
  )
  def test_max_tree_sensitivity_square_sum_every_step_part(self, max_part):
    steps, min_separation = 8, 0
    assert max_part > steps
    # If a sample will appear in every leaf node, we can infer the total
    # sensitivity by adding all the nodes.
    expected = 120
    self.assertEqual(
        expected,
        tree_aggregation_accountant._max_tree_sensitivity_square_sum(
            max_part, min_separation, steps))

  @parameterized.named_parameters(
      ('3', 3),
      ('8', 8),
      ('11', 11),
      ('19', 19),
  )
  def test_max_tree_sensitivity_square_sum_every_step_part2(self, steps):
    max_participation, min_separation = 2, 0
    # If a sample will appear twice, the worst case is to put the two nodes at
    # consecutive nodes of the deepest subtree.
    steps_bin = bin(steps)[2:]
    depth = len(steps_bin) - 1
    expected = 2 + 4 * depth
    self.assertEqual(
        expected,
        tree_aggregation_accountant._max_tree_sensitivity_square_sum(
            max_participation, min_separation, steps))

  @parameterized.named_parameters(
      ('test1', 1, 7, 8, 4),
      ('test2', 3, 3, 9, 11),
      ('test3', 3, 2, 7, 9),
      # This is an example showing worst-case sensitivity is larger than greedy
      # in "Practical and Private (Deep) Learning without Sampling or Shuffling"
      # https://arxiv.org/abs/2103.00039.
      ('test4', 8, 2, 24, 88),
  )
  def test_max_tree_sensitivity_square_sum_toy(self, max_participation,
                                               min_separation, steps, expected):
    self.assertEqual(
        expected,
        tree_aggregation_accountant._max_tree_sensitivity_square_sum(
            max_participation, min_separation, steps))

  def test_compute_gaussian_zcdp(self):
    for sigma in tf.random.uniform([5], minval=0.01, maxval=100).numpy():
      for sum_sensitivity_square in tf.random.uniform([5],
                                                      minval=0.01,
                                                      maxval=1000).numpy():
        self.assertEqual(
            tree_aggregation_accountant._compute_gaussian_rdp(
                sigma, sum_sensitivity_square, alpha=1),
            tree_aggregation_accountant._compute_gaussian_zcdp(
                sigma, sum_sensitivity_square))


if __name__ == '__main__':
  tf.test.main()

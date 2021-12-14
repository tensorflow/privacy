# Copyright 2021, Google LLC.
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
"""Tests for `tree_aggregation`."""
import math
import random
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import tree_aggregation


class ConstantValueGenerator(tree_aggregation.ValueGenerator):

  def __init__(self, constant_value):
    self.constant_value = constant_value

  def initialize(self):
    return ()

  def next(self, state):
    return self.constant_value, state


class TreeAggregatorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('total4_step1', 4, [1, 1, 2, 1], 1),
      ('total5_step1', 5, [1, 1, 2, 1, 2], 1),
      ('total6_step1', 6, [1, 1, 2, 1, 2, 2], 1),
      ('total7_step1', 7, [1, 1, 2, 1, 2, 2, 3], 1),
      ('total8_step1', 8, [1, 1, 2, 1, 2, 2, 3, 1], 1),
      ('total8_step2', 8, [2, 2, 4, 2, 4, 4, 6, 2], 2),
      ('total8_step0d5', 8, [0.5, 0.5, 1, 0.5, 1, 1, 1.5, 0.5], 0.5))
  def test_tree_sum_steps_expected(self, total_steps, expected_values,
                                   node_value):
    # Test whether `tree_aggregator` will output `expected_value` in each step
    # when `total_steps` of leaf nodes are traversed. The value of each tree
    # node is a constant `node_value` for test purpose. Note that `node_value`
    # denotes the "noise" without private values in private algorithms.
    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=ConstantValueGenerator(node_value))
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      self.assertEqual(expected_values[leaf_node_idx], val)

  @parameterized.named_parameters(
      ('total16_step1', 16, 1, 1),
      ('total17_step1', 17, 2, 1),
      ('total18_step1', 18, 2, 1),
      ('total19_step1', 19, 3, 1),
      ('total20_step0d5', 20, 1, 0.5),
      ('total21_step2', 21, 6, 2),
      ('total1024_step1', 1024, 1, 1),
      ('total1025_step1', 1025, 2, 1),
      ('total1026_step1', 1026, 2, 1),
      ('total1027_step1', 1027, 3, 1),
      ('total1028_step0d5', 1028, 1, 0.5),
      ('total1029_step2', 1029, 6, 2),
  )
  def test_tree_sum_last_step_expected(self, total_steps, expected_value,
                                       node_value):
    # Test whether `tree_aggregator` will output `expected_value` after
    # `total_steps` of leaf nodes are traversed. The value of each tree node
    # is a constant `node_value` for test purpose. Note that `node_value`
    # denotes the "noise" without private values in private algorithms.
    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=ConstantValueGenerator(node_value))
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
    self.assertEqual(expected_value, val)

  @parameterized.named_parameters(
      ('total16_step1', 16, 1, 1),
      ('total17_step1', 17, 2, 1),
      ('total18_step1', 18, 2, 1),
      ('total19_step1', 19, 3, 1),
      ('total20_step0d5', 20, 1, 0.5),
      ('total21_step2', 21, 6, 2),
      ('total1024_step1', 1024, 1, 1),
      ('total1025_step1', 1025, 2, 1),
      ('total1026_step1', 1026, 2, 1),
      ('total1027_step1', 1027, 3, 1),
      ('total1028_step0d5', 1028, 1, 0.5),
      ('total1029_step2', 1029, 6, 2),
  )
  def test_tree_sum_last_step_expected_value_fn(self, total_steps,
                                                expected_value, node_value):
    # Test no-arg function as stateless value generator.
    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=lambda: node_value)
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
    self.assertEqual(expected_value, val)

  @parameterized.named_parameters(
      ('total8_step1', 8, 1),
      ('total8_step2', 8, 2),
      ('total8_step0d5', 8, 0.5),
      ('total32_step0d5', 32, 0.5),
      ('total1024_step0d5', 1024, 0.5),
      ('total2020_step0d5', 2020, 0.5),
  )
  def test_tree_sum_steps_max(self, total_steps, node_value):
    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=ConstantValueGenerator(node_value))
    max_val = node_value * math.ceil(math.log2(total_steps))
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      self.assertLessEqual(val, max_val)

  @parameterized.named_parameters(
      ('total4_std1_d1000', 4, [1, 1, 2, 1], 1, [1000], 0.15),
      ('total4_std1_d10000', 4, [1, 1, 2, 1], 1, [10000], 0.05),
      ('total7_std1_d1000', 7, [1, 1, 2, 1, 2, 2, 3], 1, [1000], 0.15),
      ('total8_std1_d1000', 8, [1, 1, 2, 1, 2, 2, 3, 1], 1, [1000], 0.15),
      ('total8_std2_d1000', 8, [4, 4, 8, 4, 8, 8, 12, 4], 2, [1000], 0.15),
      ('total8_std0d5_d1000', 8, [0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.75, 0.25
                                 ], 0.5, [1000], 0.15))
  def test_tree_sum_noise_expected(self, total_steps, expected_variance,
                                   noise_std, variable_shape, tolerance):
    # Test whether `tree_aggregator` will output `expected_variance` (within a
    # relative `tolerance`) in each step when `total_steps` of leaf nodes are
    # traversed. Each tree node is a `variable_shape` tensor of Gaussian noise
    # with `noise_std`.
    random_generator = tree_aggregation.GaussianNoiseGenerator(
        noise_std, tf.TensorSpec(variable_shape), seed=2020)
    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=random_generator)
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      self.assertAllClose(
          math.sqrt(expected_variance[leaf_node_idx]),
          tf.math.reduce_std(val),
          rtol=tolerance)

  def test_cumsum_vector(self, total_steps=15):

    tree_aggregator = tree_aggregation.TreeAggregator(
        value_generator=ConstantValueGenerator([
            tf.ones([2, 2], dtype=tf.float32),
            tf.constant([2], dtype=tf.float32)
        ]))
    tree_aggregator_truth = tree_aggregation.TreeAggregator(
        value_generator=ConstantValueGenerator(1.))
    state = tree_aggregator.init_state()
    truth_state = tree_aggregator_truth.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      expected_val, truth_state = tree_aggregator_truth.get_cumsum_and_update(
          truth_state)
      self.assertEqual(
          tree_aggregation.get_step_idx(state),
          tree_aggregation.get_step_idx(truth_state))
      expected_result = [
          expected_val * tf.ones([2, 2], dtype=tf.float32),
          expected_val * tf.constant([2], dtype=tf.float32),
      ]
      tf.nest.map_structure(self.assertAllEqual, val, expected_result)


class EfficientTreeAggregatorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('total1_step1', 1, 1, 1.),
      ('total2_step1', 2, 4. / 3., 1.),
      ('total3_step1', 3, 4. / 3. + 1., 1.),
      ('total4_step1', 4, 12. / 7., 1.),
      ('total5_step1', 5, 12. / 7. + 1., 1.),
      ('total6_step1', 6, 12. / 7. + 4. / 3., 1.),
      ('total7_step1', 7, 12. / 7. + 4. / 3. + 1., 1.),
      ('total8_step1', 8, 32. / 15., 1.),
      ('total1024_step1', 1024, 11. / (2 - .5**10), 1.),
      ('total1025_step1', 1025, 11. / (2 - .5**10) + 1., 1.),
      ('total1026_step1', 1026, 11. / (2 - .5**10) + 4. / 3., 1.),
      ('total1027_step1', 1027, 11. / (2 - .5**10) + 4. / 3. + 1.0, 1.),
      ('total1028_step0d5', 1028, (11. / (2 - .5**10) + 12. / 7.) * .5, .5),
      ('total1029_step2', 1029, (11. / (2 - .5**10) + 12. / 7. + 1.) * 2., 2.),
  )
  def test_tree_sum_last_step_expected(self, total_steps, expected_value,
                                       step_value):
    # Test whether `tree_aggregator` will output `expected_value` after
    # `total_steps` of leaf nodes are traversed. The value of each tree node
    # is a constant `node_value` for test purpose. Note that `node_value`
    # denotes the "noise" without private values in private algorithms. The
    # `expected_value` is based on a weighting schema strongly depends on the
    # depth of the binary tree.
    tree_aggregator = tree_aggregation.EfficientTreeAggregator(
        value_generator=ConstantValueGenerator(step_value))
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
    self.assertAllClose(expected_value, val)

  @parameterized.named_parameters(
      ('total4_std1_d1000', 4, 4. / 7., 1., [1000], 0.15),
      ('total4_std1_d10000', 4, 4. / 7., 1., [10000], 0.05),
      ('total7_std1_d1000', 7, 4. / 7. + 2. / 3. + 1., 1, [1000], 0.15),
      ('total8_std1_d1000', 8, 8. / 15., 1., [1000], 0.15),
      ('total8_std2_d1000', 8, 8. / 15. * 4, 2., [1000], 0.15),
      ('total8_std0d5_d1000', 8, 8. / 15. * .25, .5, [1000], 0.15))
  def test_tree_sum_noise_expected(self, total_steps, expected_variance,
                                   noise_std, variable_shape, tolerance):
    # Test whether `tree_aggregator` will output `expected_variance` (within a
    # relative `tolerance`) after  `total_steps` of leaf nodes are traversed.
    # Each tree node is a `variable_shape` tensor of Gaussian noise with
    # `noise_std`. Note that the variance of a tree node is smaller than
    # the given vanilla node `noise_std` because of the update rule of
    # `EfficientTreeAggregator`.
    random_generator = tree_aggregation.GaussianNoiseGenerator(
        noise_std, tf.TensorSpec(variable_shape), seed=2020)
    tree_aggregator = tree_aggregation.EfficientTreeAggregator(
        value_generator=random_generator)
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
    self.assertAllClose(
        math.sqrt(expected_variance), tf.math.reduce_std(val), rtol=tolerance)

  @parameterized.named_parameters(
      ('total4_std1_d1000', 4, 1., [1000], 1e-6),
      ('total30_std2_d1000', 30, 2, [1000], 1e-6),
      ('total32_std0d5_d1000', 32, .5, [1000], 1e-6),
      ('total60_std1_d1000', 60, 1, [1000], 1e-6),
  )
  def test_tree_sum_noise_efficient(self, total_steps, noise_std,
                                    variable_shape, tolerance):
    # Test the variance returned by `EfficientTreeAggregator` is smaller than
    # `TreeAggregator` (within a relative `tolerance`) after `total_steps` of
    # leaf nodes are traversed. Each tree node is a `variable_shape` tensor of
    # Gaussian noise with `noise_std`. A small `tolerance` is used for numerical
    # stability, `tolerance==0` means `EfficientTreeAggregator` is strictly
    # better than `TreeAggregator` for reducing variance.
    random_generator = tree_aggregation.GaussianNoiseGenerator(
        noise_std, tf.TensorSpec(variable_shape))
    tree_aggregator = tree_aggregation.EfficientTreeAggregator(
        value_generator=random_generator)
    tree_aggregator_baseline = tree_aggregation.TreeAggregator(
        value_generator=random_generator)

    state = tree_aggregator.init_state()
    state_baseline = tree_aggregator_baseline.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      val_baseline, state_baseline = tree_aggregator_baseline.get_cumsum_and_update(
          state_baseline)
    self.assertLess(
        tf.math.reduce_variance(val),
        (1 + tolerance) * tf.math.reduce_variance(val_baseline))

  def test_cumsum_vector(self, total_steps=15):

    tree_aggregator = tree_aggregation.EfficientTreeAggregator(
        value_generator=ConstantValueGenerator([
            tf.ones([2, 2], dtype=tf.float32),
            tf.constant([2], dtype=tf.float32)
        ]))
    tree_aggregator_truth = tree_aggregation.EfficientTreeAggregator(
        value_generator=ConstantValueGenerator(1.))
    state = tree_aggregator.init_state()
    truth_state = tree_aggregator_truth.init_state()
    for leaf_node_idx in range(total_steps):
      self.assertEqual(leaf_node_idx, tree_aggregation.get_step_idx(state))
      val, state = tree_aggregator.get_cumsum_and_update(state)
      expected_val, truth_state = tree_aggregator_truth.get_cumsum_and_update(
          truth_state)
      self.assertEqual(
          tree_aggregation.get_step_idx(state),
          tree_aggregation.get_step_idx(truth_state))
      expected_result = [
          expected_val * tf.ones([2, 2], dtype=tf.float32),
          expected_val * tf.constant([2], dtype=tf.float32),
      ]
      tf.nest.map_structure(self.assertAllClose, val, expected_result)


class GaussianNoiseGeneratorTest(tf.test.TestCase, parameterized.TestCase):

  def assertStateEqual(self, state1, state2):
    for s1, s2 in zip(tf.nest.flatten(state1), tf.nest.flatten(state2)):
      self.assertAllEqual(s1, s2)

  def test_random_generator_tf(self,
                               noise_mean=1.0,
                               noise_std=1.0,
                               samples=1000,
                               tolerance=0.15):
    g = tree_aggregation.GaussianNoiseGenerator(
        noise_std, specs=tf.TensorSpec([]), seed=2020)
    gstate = g.initialize()

    @tf.function
    def return_noise(state):
      value, state = g.next(state)
      return noise_mean + value, state

    noise_values = []
    for _ in range(samples):
      value, gstate = return_noise(gstate)
      noise_values.append(value)
    noise_values = tf.stack(noise_values)
    self.assertAllClose(
        [tf.math.reduce_mean(noise_values),
         tf.math.reduce_std(noise_values)], [noise_mean, noise_std],
        rtol=tolerance)

  def test_seed_state(self, seed=1, steps=32, noise_std=0.1):
    g = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=tf.TensorSpec([]), seed=seed)
    gstate = g.initialize()
    g2 = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=tf.TensorSpec([]), seed=seed)
    gstate2 = g.initialize()
    self.assertStateEqual(gstate, gstate2)
    for _ in range(steps):
      value, gstate = g.next(gstate)
      value2, gstate2 = g2.next(gstate2)
      self.assertAllEqual(value, value2)
      self.assertStateEqual(gstate, gstate2)

  def test_seed_state_nondeterministic(self, steps=32, noise_std=0.1):
    g = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=tf.TensorSpec([]))
    gstate = g.initialize()
    g2 = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=tf.TensorSpec([]))
    gstate2 = g2.initialize()
    self.assertNotAllEqual(gstate.seeds, gstate2.seeds)
    for _ in range(steps):
      value, gstate = g.next(gstate)
      value2, gstate2 = g2.next(gstate2)
      self.assertNotAllEqual(value, value2)
      self.assertNotAllEqual(gstate.seeds, gstate2.seeds)

  def test_seed_state_structure(self, seed=1, steps=32, noise_std=0.1):
    specs = [tf.TensorSpec([]), tf.TensorSpec([1]), tf.TensorSpec([2, 2])]
    g = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=specs, seed=seed)
    gstate = g.initialize()
    g2 = tree_aggregation.GaussianNoiseGenerator(
        noise_std=noise_std, specs=specs, seed=seed)
    gstate2 = g2.initialize()
    self.assertStateEqual(gstate, gstate2)
    for _ in range(steps):
      value, gstate = g.next(gstate)
      value2, gstate2 = g2.next(gstate2)
      self.assertAllClose(value, value2)
      self.assertStateEqual(gstate, gstate2)

  @parameterized.named_parameters(
      ('increase', range(10), 1),
      ('decrease', range(30, 20, -2), 2),
      ('flat', [3.0] * 5, 1),
      ('small', [0.1**x for x in range(4)], 4),
      ('random', [random.uniform(1, 10) for _ in range(5)], 4),
  )
  def test_adaptive_stddev(self, stddev_list, reset_frequency):
    # The stddev estimation follows a chi distribution. The confidence for
    # `sample_num` samples should be high, and we use a relatively large
    # tolerance to guard the numerical stability for small stddev values.
    sample_num, tolerance = 10000, 0.05
    g = tree_aggregation.GaussianNoiseGenerator(
        noise_std=1., specs=tf.TensorSpec([sample_num]), seed=2021)
    gstate = g.initialize()
    for stddev in stddev_list:
      gstate = g.make_state(gstate.seeds, tf.constant(stddev, dtype=tf.float32))
      for _ in range(reset_frequency):
        prev_gstate = gstate
        value, gstate = g.next(gstate)
        print(tf.math.reduce_std(value), stddev)
        self.assertAllClose(tf.math.reduce_std(value), stddev, rtol=tolerance)
        self.assertNotAllEqual(gstate.seeds, prev_gstate.seeds)


if __name__ == '__main__':
  tf.test.main()

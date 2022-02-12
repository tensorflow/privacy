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
"""Tests for advanced_mia."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia


class TestScoreOffset(parameterized.TestCase):
  """Tests compute_score_offset."""

  def setUp(self):
    super().setUp()
    self.stat_target = np.array([[-0.1, 0.1, 0], [0, 0, 27],
                                 [0, 0, 0]])  # 3 samples with 3 augmentations
    self.stat_in = [
        np.array([[1, 2, -3.]]),  # 1 shadow
        np.array([[-2., 4, 6], [0, 0, 0], [5, -7, -9]]),  # 3 shadow
        np.empty((0, 3))
    ]  # no shadow
    self.stat_out = [-s + 10 for s in self.stat_in]

  @parameterized.named_parameters(
      ('both_mean', 'both', 'mean', np.array([-5., 4., -5.])),
      ('both_median', 'both', 'median', np.array([-5., 4., -5.])),
      ('in_median', 'in', 'median', np.array([0., 9., 0.])),
      ('out_median', 'out', 'median', np.array([-10., -1., -10.])),
      ('in_mean', 'in', 'mean', np.array([0, 28. / 3, 1. / 6])),
      ('out_mean', 'out', 'mean', np.array([-10, -4. / 3, -61. / 6])))
  def test_compute_score_offset(self, option, median_or_mean, expected):
    scores = amia.compute_score_offset(self.stat_target, self.stat_in,
                                       self.stat_out, option, median_or_mean)
    np.testing.assert_allclose(scores, expected, atol=1e-7)
    # If `option` is "in" (resp. out), test with `stat_out` (resp. `stat_out`)
    # setting to empty list.
    if option == 'in':
      scores = amia.compute_score_offset(self.stat_target, self.stat_in, [],
                                         option, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)
    elif option == 'out':
      scores = amia.compute_score_offset(self.stat_target, [], self.stat_out,
                                         option, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)


class TestLiRA(parameterized.TestCase):
  """Tests compute_score_lira."""

  @parameterized.named_parameters(
      ('in_median', 'in', False, 'median',
       np.array([1.41893853, 1.41893853, 3.72152363, 2.72537178])),
      ('in_mean', 'in', False, 'mean',
       np.array([1.41893853, 1.41893853, 3.72152363, 2.72537178])),
      ('out_median', 'out', False, 'median',
       -np.array([1.41893853, 1.41893853, 3.72152363, 2.72537178])),
      ('out_mean', 'out', False, 'mean',
       -np.array([1.41893853, 1.41893853, 3.72152363, 2.72537178])),
      ('in_median_fix', 'in', True, 'median',
       np.array([2.69682468, 2.69682468, 4.15270703, 2.87983121])),
      ('in_mean_fix', 'in', True, 'mean',
       np.array([2.69682468, 2.69682468, 4.15270703, 2.87983121])),
      ('out_median_fix', 'out', True, 'median',
       -np.array([2.69682468, 2.69682468, 4.15270703, 2.87983121])),
      ('out_mean_fix', 'out', True, 'mean',
       -np.array([2.69682468, 2.69682468, 4.15270703, 2.87983121])),
      ('both_median_fix', 'both', True, 'median', np.array([0, 0, 0, 0.])),
      ('both_mean_fix', 'both', True, 'mean', np.array([0, 0, 0, 0.])),
      ('both_median', 'both', False, 'median', np.array([0, 0, 0, 0.])),
      ('both_mean', 'both', False, 'mean', np.array([0, 0, 0, 0.])),
  )
  def test_with_one_augmentation(self, option, fix_variance, median_or_mean,
                                 expected):
    stat_target = np.array([[1.], [0.], [0.], [0.]])
    stat_in = [
        np.array([[-1], [1.]]),
        np.array([[0], [2.]]),
        np.array([[0], [20.]]),
        np.empty((0, 1))
    ]
    stat_out = [-s for s in stat_in]

    scores = amia.compute_score_lira(stat_target, stat_in, stat_out, option,
                                     fix_variance, median_or_mean)
    np.testing.assert_allclose(scores, expected, atol=1e-7)
    # If `option` is "in" (resp. out), test with `stat_out` (resp. `stat_out`)
    # setting to empty list.
    if option == 'in':
      scores = amia.compute_score_lira(stat_target, stat_in, [], option,
                                       fix_variance, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)
    elif option == 'out':
      scores = amia.compute_score_lira(stat_target, [], stat_out, option,
                                       fix_variance, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)

  @parameterized.named_parameters(
      ('in_median', 'in', False, 'median', 2.57023108),
      ('in_mean', 'in', False, 'mean', 2.57023108),
      ('out_median', 'out', False, 'median', -2.57023108),
      ('out_mean', 'out', False, 'mean', -2.57023108),
      ('both_median', 'both', False, 'median', 0),
      ('both_mean', 'both', False, 'mean', 0))
  def test_two_augmentations(self, option, fix_variance, median_or_mean,
                             expected):
    stat_target = np.array([[1., 0.]])
    stat_in = [np.array([[-1, 0], [1., 20]])]
    stat_out = [-s for s in stat_in]

    scores = amia.compute_score_lira(stat_target, stat_in, stat_out, option,
                                     fix_variance, median_or_mean)
    np.testing.assert_allclose(scores, expected, atol=1e-7)
    # If `option` is "in" (resp. out), test with `stat_out` (resp. `stat_out`)
    # setting to empty list.
    if option == 'in':
      scores = amia.compute_score_lira(stat_target, stat_in, [], option,
                                       fix_variance, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)
    elif option == 'out':
      scores = amia.compute_score_lira(stat_target, [], stat_out, option,
                                       fix_variance, median_or_mean)
      np.testing.assert_allclose(scores, expected, atol=1e-7)


class TestLogitProbConversion(absltest.TestCase):
  """Test convert_logit_to_prob."""

  def test_convert_logit_to_prob(self):
    """Test convert_logit_to_prob."""
    logit = np.array([[10, -1, 0.], [-10, 0, -11]])
    prob = amia.convert_logit_to_prob(logit)
    expected = np.array([[9.99937902e-01, 1.67006637e-05, 4.53971105e-05],
                         [4.53971105e-05, 9.99937902e-01, 1.67006637e-05]])
    np.testing.assert_allclose(prob, expected, atol=1e-5)


class TestCalculateStatistic(absltest.TestCase):
  """Test calculate_statistic."""

  def test_calculate_statistic_logit(self):
    """Test calculate_statistic with input as logit."""
    is_logits = True
    logit = np.array([[1, 2, -3.], [-1, 1, 0]])
    # expected probability vector
    # array([[0.26762315, 0.72747516, 0.00490169],
    #        [0.09003057, 0.66524096, 0.24472847]])
    labels = np.array([1, 2])

    stat = amia.calculate_statistic(logit, labels, is_logits, 'conf with prob')
    np.testing.assert_allclose(stat, np.array([0.72747516, 0.24472847]))

    stat = amia.calculate_statistic(logit, labels, is_logits, 'xe')
    np.testing.assert_allclose(stat, np.array([0.31817543, 1.40760596]))

    stat = amia.calculate_statistic(logit, labels, is_logits, 'logit')
    np.testing.assert_allclose(stat, np.array([0.98185009, -1.12692802]))

    stat = amia.calculate_statistic(logit, labels, is_logits, 'conf with logit')
    np.testing.assert_allclose(stat, np.array([2, 0.]))

    stat = amia.calculate_statistic(logit, labels, is_logits, 'hinge')
    np.testing.assert_allclose(stat, np.array([1, -1.]))

  def test_calculate_statistic_prob(self):
    """Test calculate_statistic with input as probability vector."""
    is_logits = False
    prob = np.array([[0.1, 0.85, 0.05], [0.1, 0.5, 0.4]])
    labels = np.array([1, 2])

    stat = amia.calculate_statistic(prob, labels, is_logits, 'conf with prob')
    np.testing.assert_allclose(stat, np.array([0.85, 0.4]))

    stat = amia.calculate_statistic(prob, labels, is_logits, 'xe')
    np.testing.assert_allclose(stat, np.array([0.16251893, 0.91629073]))

    stat = amia.calculate_statistic(prob, labels, is_logits, 'logit')
    np.testing.assert_allclose(stat, np.array([1.73460106, -0.40546511]))

    np.testing.assert_raises(ValueError, amia.calculate_statistic, prob, labels,
                             is_logits, 'conf with logit')
    np.testing.assert_raises(ValueError, amia.calculate_statistic, prob, labels,
                             is_logits, 'hinge')


if __name__ == '__main__':
  absltest.main()

# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_privacy.privacy.privacy_tests import epsilon_lower_bound as elb


class TPFPTest(parameterized.TestCase):

  def test_tp_fp_given_thresholds(self):
    pos_scores = np.array([9, 4, 5, 0.])
    neg_scores = np.array([3, 8, 6, 1, 2, 7])
    thresholds = np.array([9.5, 8.5, 5.5, 3.5, 0.5, -1])
    expected_tp = np.array([0, 1, 1, 3, 3, 4])
    expected_fp = np.array([0, 0, 3, 3, 6, 6])
    tp, fp = elb._get_tp_fp_for_thresholds(pos_scores, neg_scores, thresholds)
    np.testing.assert_array_equal(tp, expected_tp)
    np.testing.assert_array_equal(fp, expected_fp)

  def test_tp_fp_all_thresholds(self):
    pos_scores = np.array([9, 4, 5, 0])
    neg_scores = np.array([3, 8, 6, 1, 2, 7.])
    expected_tp = np.array([1, 1, 1, 1, 2, 3, 3, 3, 3, 4])
    expected_fp = np.array([0, 1, 2, 3, 3, 3, 4, 5, 6, 6])
    tp, fp = elb._get_tp_fp_for_thresholds(pos_scores, neg_scores)
    np.testing.assert_array_equal(tp, expected_tp)
    np.testing.assert_array_equal(fp, expected_fp)


class RatioBoundTest(parameterized.TestCase):

  # For every method except for clopper pearson, expected results are from
  # https://CRAN.R-project.org/package=asbio `ci.prat`.
  # For clopper pearson, test case is from https://arxiv.org/pdf/2006.07709.pdf.

  @parameterized.parameters(
      (473, 511, 755, 950, elb.BoundMethod.KATZ_LOG, 1.01166194),
      (473, 511, 755, 950, elb.BoundMethod.ADJUSTED_LOG, 1.01189079),
      (473, 511, 755, 950, elb.BoundMethod.BAILEY, 1.01148934),
      (473, 511, 755, 950, elb.BoundMethod.INV_SINH, 1.01167669),
      (2, 697, 213, 313, elb.BoundMethod.KATZ_LOG, 0.00220470),
      (2, 697, 213, 313, elb.BoundMethod.ADJUSTED_LOG, 0.00310880),
      (2, 697, 213, 313, elb.BoundMethod.BAILEY, 0.00162322),
      (2, 697, 213, 313, elb.BoundMethod.INV_SINH, 0.00233503),
      (1000, 10, 10, 10000, elb.BoundMethod.KATZ_LOG, 589.27335593),
      (1000, 10, 10, 10000, elb.BoundMethod.ADJUSTED_LOG, 568.32659563),
      (1000, 10, 10, 10000, elb.BoundMethod.BAILEY, 613.58766147),
      (1000, 10, 10, 10000, elb.BoundMethod.INV_SINH, 592.63264572),
  )
  def test_bound_scalar(self, tp, fn, fp, tn, method, expected_value):
    rb = elb.RatioBound(tp, fp, tp + fn, fp + tn, 0.05)
    res = rb.compute_bound(method)
    self.assertAlmostEqual(res, expected_value)

  @parameterized.parameters(
      (0, 511, 755, 950, elb.BoundMethod.KATZ_LOG, 0.),
      (0, 511, 755, 950, elb.BoundMethod.ADJUSTED_LOG, 0.00021568),
      (0, 511, 755, 950, elb.BoundMethod.BAILEY, 0.),
      (0, 511, 755, 950, elb.BoundMethod.INV_SINH, 0.),
      (2, 697, 0, 313, elb.BoundMethod.KATZ_LOG, 0.13325535),
      (2, 697, 0, 313, elb.BoundMethod.ADJUSTED_LOG, 0.17571862),
      (2, 697, 0, 313, elb.BoundMethod.BAILEY, 0.18481008),
      (2, 697, 0, 313, elb.BoundMethod.INV_SINH, 0.08081083),
      (0, 10, 0, 10000, elb.BoundMethod.KATZ_LOG, 0.),
      (0, 10, 0, 10000, elb.BoundMethod.ADJUSTED_LOG, 0.),
      (0, 10, 0, 10000, elb.BoundMethod.BAILEY, 0.),
      (0, 10, 0, 10000, elb.BoundMethod.INV_SINH, 0.),
  )
  def test_bound_scalar_with_0(self, tp, fn, fp, tn, method, expected_value):
    rb = elb.RatioBound(tp, fp, tp + fn, fp + tn, 0.05)
    res = rb.compute_bound(method)
    self.assertAlmostEqual(res, expected_value)

  @parameterized.parameters(
      (473, 0, 755, 950, elb.BoundMethod.KATZ_LOG, 2.15959024),
      (473, 0, 755, 950, elb.BoundMethod.ADJUSTED_LOG, 2.15883995),
      (473, 0, 755, 950, elb.BoundMethod.BAILEY, 2.15787013),
      (473, 0, 755, 950, elb.BoundMethod.INV_SINH, 2.15959827),
      (2, 697, 213, 0, elb.BoundMethod.KATZ_LOG, 0.00089568),
      (2, 697, 213, 0, elb.BoundMethod.ADJUSTED_LOG, 0.00126522),
      (2, 697, 213, 0, elb.BoundMethod.BAILEY, 0.00066016),
      (2, 697, 213, 0, elb.BoundMethod.INV_SINH, 0.00094821),
      (1000, 0, 10, 0, elb.BoundMethod.KATZ_LOG, 0.93375356),
      (1000, 0, 10, 0, elb.BoundMethod.ADJUSTED_LOG, 0.93686007),
      (1000, 0, 10, 0, elb.BoundMethod.BAILEY, 0.93591485),
      (1000, 0, 10, 0, elb.BoundMethod.INV_SINH, 0.93381958),
  )
  def test_bound_scalar_with_large(self, tp, fn, fp, tn, method,
                                   expected_value):
    rb = elb.RatioBound(tp, fp, tp + fn, fp + tn, 0.05)
    res = rb.compute_bound(method)
    self.assertAlmostEqual(res, expected_value)

  @parameterized.parameters(
      (elb.BoundMethod.KATZ_LOG,
       np.array([1.71127264, 0., 2.21549347, 0.99840355])),
      (elb.BoundMethod.ADJUSTED_LOG,
       np.array([1.71070821, 0.00388765, 2.33237436, 0.99840433])),
      (elb.BoundMethod.BAILEY, np.array(
          [1.71182320, 0., 3.86813751, 0.99840348])),
      (elb.BoundMethod.INV_SINH,
       np.array([1.71128185, 0., 1.51711034, 0.99840355])),
  )
  def test_bound_array(self, method, expected_values):
    pos_size, neg_size = 1000, 1200
    tp, fp = np.array([900, 0, 10, 1000]), np.array([600, 14, 0, 1200])
    rb = elb.RatioBound(tp, fp, pos_size, neg_size, 0.05)
    res = rb.compute_bound(method)
    np.testing.assert_allclose(res, expected_values, atol=1e-7)

    # Also test when the input is 1-element array
    rb = elb.RatioBound(tp[:1], fp[:1], pos_size, neg_size, 0.05)
    res = rb.compute_bound(method)
    self.assertLen(res, 1)
    np.testing.assert_allclose(res, expected_values[:1], atol=1e-7)

  def test_bound_scalar_clopper_pearson(self):
    tp, fp, tn, fn = 500, 0, 500, 0
    rb = elb.RatioBound(tp, fp, tp + fn, fp + tn, 0.01)
    res = rb.compute_bound(elb.BoundMethod.CLOPPER_PEARSON)
    self.assertAlmostEqual(np.log(res), 4.54, places=2)

  def test_bounds_scalar(self):
    rb = elb.RatioBound(500, 0, 500, 500, 0.01)
    # Expected result except for clopper pearson.
    expected_res = {
        elb.BoundMethod.KATZ_LOG: 37.31696140,
        elb.BoundMethod.ADJUSTED_LOG: 37.35421695,
        elb.BoundMethod.BAILEY: 108.47420518,
        elb.BoundMethod.INV_SINH: 35.46128691
    }
    res = rb.compute_bounds()
    # For clopper pearson we only have this test. So we handle it separately.
    self.assertAlmostEqual(
        np.log(res[elb.BoundMethod.CLOPPER_PEARSON]), 4.54, places=2)
    del res[elb.BoundMethod.CLOPPER_PEARSON]
    self.assertEqual(res.keys(), expected_res.keys())
    np.testing.assert_almost_equal([res[k] for k in res],
                                   [expected_res[k] for k in res])
    # Specify methods to use
    methods = set([elb.BoundMethod.KATZ_LOG, elb.BoundMethod.INV_SINH])
    res = rb.compute_bounds(methods)
    self.assertEqual(set(res.keys()), methods)
    np.testing.assert_almost_equal([res[k] for k in res],
                                   [expected_res[k] for k in res])


class EpsilonLowerBoundTest(parameterized.TestCase):

  def test_epsilon_bound(self):
    pos_scores = np.array([9, 4, 5, 0.])
    neg_scores = np.array([3, 8, 6, 1, 2, 7])
    thresholds = np.array([9.5, 8.5, 5.5, 0.5])
    alpha = 0.05
    method = elb.BoundMethod.ADJUSTED_LOG
    # Therefore,
    #   pos_size = 4, neg_size = 6
    #   tp = [0, 1, 1, 3]
    #   fp = [0, 0, 3, 6]
    #   fn = [4, 3, 3, 1]
    #   tn = [6, 6, 3, 0]
    # The expected epsilon bounds for the four ratios:
    tpr_fpr = [-1.02310327, -1.72826789, -0.66577858]
    tnr_fnr = [-0.29368152, -0.16314973, -1.09474300, -3.95577741]
    fpr_tpr = [-3.95577741, -0.76912173, -0.16314973]
    fnr_tnr = [-0.36916217, -0.66577858, -0.35929344, -1.02310327]
    expected_one_sided = np.sort(tpr_fpr + tnr_fnr)[::-1]
    expected_two_sided = np.sort(tpr_fpr + tnr_fnr + fpr_tpr + fnr_tnr)[::-1]

    common_kwargs = {
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
        'alpha': alpha,
        'thresholds': thresholds
    }

    # one-sided
    lb = elb.EpsilonLowerBound(two_sided_threshold=False, **common_kwargs)
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method), expected_one_sided)

    # two-sided
    lb = elb.EpsilonLowerBound(two_sided_threshold=True, **common_kwargs)
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method), expected_two_sided)

    # test for top-k
    k = 5
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method, k), expected_two_sided[:k])
    k = 100
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method, k), expected_two_sided)

  def test_epsilon_bounds(self):
    pos_scores = np.array([9, 4, 5, 0.])
    neg_scores = np.array([3, 8, 6, 1, 2, 7])
    thresholds = np.array([9.5, 8.5, 5.5, 0.5])
    alpha = 0.05
    # Therefore,
    #   tp = [0, 1, 1, 3]
    #   fp = [0, 0, 3, 6]
    #   fn = [4, 3, 3, 1]
    #   tn = [6, 6, 3, 0]
    # The expected epsilon bounds for using tpr / fpr, tnr / fnr:
    expected = {
        elb.BoundMethod.ADJUSTED_LOG:
            np.sort([-1.02310327, -1.72826789, -0.66577858] +
                    [-0.29368152, -0.16314973, -1.09474300, -3.95577741])[::-1],
        elb.BoundMethod.INV_SINH:
            np.sort([-2.05961264, -2.13876684, -0.75815925] +
                    [-0.32235626, -0.18279510, -1.20631805])[::-1]
    }
    lb = elb.EpsilonLowerBound(
        pos_scores,
        neg_scores,
        alpha,
        two_sided_threshold=False,
        thresholds=thresholds)
    res = lb.compute_epsilon_lower_bounds(expected.keys())
    self.assertEqual(res.keys(), expected.keys())
    for method in expected:
      np.testing.assert_almost_equal(res[method], expected[method])
    # test for top-k
    k = 5
    res = lb.compute_epsilon_lower_bounds(expected.keys(), k)
    self.assertEqual(res.keys(), expected.keys())
    for method in expected:
      np.testing.assert_almost_equal(res[method], expected[method][:k])

  def test_epsilon_bound_clopper_pearson(self):
    # Try to create tp, fp, tn, fn = 500, 0, 500, 0
    pos_scores = np.ones(500)
    neg_scores = np.zeros(500)
    thresholds = np.array([0.5])
    alpha = 0.01
    expected_eps = 4.54
    method = elb.BoundMethod.CLOPPER_PEARSON

    # one-sided
    lb = elb.EpsilonLowerBound(
        pos_scores,
        neg_scores,
        alpha,
        thresholds=thresholds,
        two_sided_threshold=False)
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method), [expected_eps, expected_eps],
        decimal=2)
    # two-sided. fpr / tpr = fnr / tnr = 0
    lb = elb.EpsilonLowerBound(
        pos_scores,
        neg_scores,
        alpha,
        thresholds=thresholds,
        two_sided_threshold=True)
    np.testing.assert_almost_equal(
        lb.compute_epsilon_lower_bound(method), [expected_eps, expected_eps],
        decimal=2)


if __name__ == '__main__':
  absltest.main()

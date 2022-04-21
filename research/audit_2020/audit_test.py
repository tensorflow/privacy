# Copyright 2021, The TensorFlow Authors.
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

"""Tests for audit.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import audit

def dummy_train_and_score_function(dataset):
  del dataset
  return 0

def get_auditor():
  poisoning = {}
  datasets = (np.zeros((5, 2)), np.zeros(5)), (np.zeros((5, 2)), np.zeros(5))
  poisoning["data"] = datasets
  poisoning["pois"] = (datasets[0][0][0], datasets[0][1][0])
  auditor = audit.AuditAttack(datasets[0][0], datasets[0][1],
                              dummy_train_and_score_function)
  auditor.poisoning = poisoning

  return auditor


class AuditParameterizedTest(parameterized.TestCase):
  """Class to test parameterized audit.py functions."""
  @parameterized.named_parameters(
      ('Test0', np.ones(500), np.zeros(500), 0.5, 0.01, 1,
       (4.541915810224092, 0.9894593118113243)),
      ('Test1', np.ones(500), np.zeros(500), 0.5, 0.01, 2,
       (2.27095790511, 0.9894593118113243)),
      ('Test2', np.ones(500), np.ones(500), 0.5, 0.01, 1,
       (0, 0))
  )

  def test_compute_epsilon_and_acc(self, poison_scores, unpois_scores,
                                   threshold, pois_ct, alpha, expected_res):
    expected_eps, expected_acc = expected_res
    computed_res = audit.compute_epsilon_and_acc(poison_scores, unpois_scores,
                                                 threshold, pois_ct, alpha)
    computed_eps, computed_acc = computed_res
    self.assertAlmostEqual(computed_eps, expected_eps)
    self.assertAlmostEqual(computed_acc, expected_acc)

  @parameterized.named_parameters(
      ('Test0', [1]*500, [0]*250 + [.5]*250, 1, 0.01, .5,
       (.5, 4.541915810224092, 0.9894593118113243)),
      ('Test1', [1]*500, [0]*250 + [.5]*250, 1, 0.01, None,
       (.5, 4.541915810224092, 0.9894593118113243)),
      ('Test2', [1]*500, [0]*500, 2, 0.01, .5,
       (.5, 2.27095790511, 0.9894593118113243)),
  )

  def test_compute_results(self, poison_scores, unpois_scores, pois_ct,
                           alpha, threshold, expected_res):
    expected_thresh, expected_eps, expected_acc = expected_res
    computed_res = audit.compute_results(poison_scores, unpois_scores,
                                         pois_ct, alpha, threshold)
    computed_thresh, computed_eps, computed_acc = computed_res
    self.assertAlmostEqual(computed_thresh, expected_thresh)
    self.assertAlmostEqual(computed_eps, expected_eps)
    self.assertAlmostEqual(computed_acc, expected_acc)


class AuditAttackTest(absltest.TestCase):
  """Nonparameterized audit.py test class."""
  def test_run_experiments(self):
    auditor = get_auditor()
    pois, unpois = auditor.run_experiments(100)
    expected = [0]*100
    self.assertListEqual(pois, expected)
    self.assertListEqual(unpois, expected)



if __name__ == '__main__':
  absltest.main()

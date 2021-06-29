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

# Lint as: python3
"""Tests for tensorflow_privacy.privacy.secret_sharer.exposures."""

from absl.testing import absltest

import numpy as np
from scipy.stats import skewnorm

from tensorflow_privacy.privacy.privacy_tests.secret_sharer.exposures import compute_exposure_extrapolation
from tensorflow_privacy.privacy.privacy_tests.secret_sharer.exposures import compute_exposure_interpolation


class UtilsTest(absltest.TestCase):

  def __init__(self, methodname):
    """Initialize the test class."""
    super().__init__(methodname)

  def test_exposure_interpolation(self):
    """Test exposure by interpolation."""
    perplexities = {1: [0, 0.1],  # smallest perplexities
                    2: [20.0],  # largest perplexities
                    5: [3.5]}  # rank = 4
    perplexities_reference = [float(x) for x in range(1, 17)]
    exposures = compute_exposure_interpolation(perplexities,
                                               perplexities_reference)
    num_perplexities_reference = len(perplexities_reference)
    exposure_largest = np.log2(num_perplexities_reference)
    exposure_smallest = np.log2(num_perplexities_reference) - np.log2(
        num_perplexities_reference + 1)
    expected_exposures = {
        1: np.array([exposure_largest] * 2),
        2: np.array([exposure_smallest]),
        5: np.array([np.log2(num_perplexities_reference) - np.log2(4)])}

    self.assertEqual(exposures.keys(), expected_exposures.keys())
    for r in exposures.keys():
      np.testing.assert_almost_equal(exposures[r], exposures[r])

  def test_exposure_extrapolation(self):
    parameters = (4, 0, 1)
    perplexities = {1: skewnorm.rvs(*parameters, size=(2,)),
                    10: skewnorm.rvs(*parameters, size=(5,))}
    perplexities_reference = skewnorm.rvs(*parameters, size=(10000,))
    exposures = compute_exposure_extrapolation(perplexities,
                                               perplexities_reference)
    fitted_parameters = skewnorm.fit(perplexities_reference)

    self.assertEqual(exposures.keys(), perplexities.keys())
    for r in exposures.keys():
      np.testing.assert_almost_equal(
          exposures[r],
          -np.log2(skewnorm.cdf(perplexities[r], *fitted_parameters)))


if __name__ == '__main__':
  absltest.main()

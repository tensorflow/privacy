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

"""Measuring exposure for secret sharer attack."""

from typing import Dict, List
import numpy as np
from scipy.stats import skewnorm


def compute_exposure_interpolation(
    perplexities: Dict[int, List[float]],
    perplexities_reference: List[float]) -> Dict[int, List[float]]:
  """Get exposure using interpolation.

  Args:
    perplexities: a dictionary, key is number of secret repetitions,
                  value is a list of perplexities
    perplexities_reference: a list, perplexities of the random sequences that
                            did not appear in the training data
  Returns:
    The exposure of every secret measured using interpolation (not necessarily
    in the same order as the input)
  """
  repetitions = list(perplexities.keys())
  # Concatenate all perplexities, including those for references
  perplexities_concat = np.concatenate([perplexities[r] for r in repetitions]
                                       + [perplexities_reference])
  # Concatenate the number of repetitions for each secret
  repetitions_concat = np.concatenate(
      [[r] * len(perplexities[r]) for r in repetitions]
      + [[0] * len(perplexities_reference)])

  # Sort the repetition list according to the corresponding perplexity
  idx = np.argsort(perplexities_concat)
  repetitions_concat = repetitions_concat[idx]

  # In the sorted repetition list, if there are m examples with repetition 0
  # (does not appear in training) in front of an example, then its rank is
  # (m + 1). To get the number of examples with repetition 0 in front of
  # any example, we use the cummulative sum of the indicator vecotr
  # (repetitions_concat == 0).
  cum_sum = np.cumsum(repetitions_concat == 0)
  ranks = {r: cum_sum[repetitions_concat == r] + 1 for r in repetitions}
  exposures = {r: np.log2(len(perplexities_reference)) - np.log2(ranks[r])
               for r in repetitions}
  return exposures


def compute_exposure_extrapolation(
    perplexities: Dict[int, List[float]],
    perplexities_reference: List[float]) -> Dict[int, List[float]]:
  """Get exposure using extrapolation.

  Args:
    perplexities: a dictionary, key is number of secret repetitions,
                  value is a list of perplexities
    perplexities_reference: a list, perplexities of the random sequences that
                            did not appear in the training data
  Returns:
    The exposure of every secret measured using extrapolation
  """
  # Fit a skew normal distribution using the perplexities of the references
  snormal_param = skewnorm.fit(perplexities_reference)

  # Estimate exposure using the fitted distribution
  exposures = {r: -np.log2(skewnorm.cdf(perplexities[r], *snormal_param))
               for r in perplexities.keys()}
  return exposures

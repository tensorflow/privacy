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

import logging
from typing import Mapping, Sequence, TypeVar

import numpy as np
from scipy import stats


_KT = TypeVar('_KT')


def compute_exposure_interpolation(
    perplexities: Mapping[_KT, Sequence[float]],
    perplexities_reference: Sequence[float]) -> Mapping[_KT, Sequence[float]]:
  """Gets exposure using interpolation.

  Args:
    perplexities: a `Mapping` where the key is an identifier for the secrets
      set, e.g. number of secret repetitions, and the value is an iterable of
      perplexities.
    perplexities_reference: perplexities of the random sequences that did not
      appear in the training data.

  Returns:
    The exposure of every secret measured using interpolation (not necessarily
    in the same order as the input), keyed in the same way as perplexities.
  """
  logging.info(
      'Will compute exposure (with interpolation) for '
      'splits %s with %s examples using %s references.',
      str(perplexities.keys()), str([len(p) for p in perplexities.values()]),
      len(perplexities_reference))

  # Get the keys in some fixed order which will be used internally only
  # further down.
  keys = list(perplexities)
  # Concatenate all perplexities, including those from `perplexities_reference`.
  # Add another dimension indicating which set the perplexity is from: -1 for
  # reference, {0, ..., len(perplexities)} for secrets
  perplexities_concat = [(p, -1) for p in perplexities_reference]
  for i, k in enumerate(keys):
    perplexities_concat.extend((p, i) for p in perplexities[k])

  # Get the indices list sorted according to the corresponding perplexity,
  # in case of tie, keep the reference before the secret
  indices_concat = np.fromiter((i for _, i in sorted(perplexities_concat)),
                               dtype=int)

  # In the sorted indices list, if there are m examples with index -1
  # (from the reference set) in front of an example, then its rank is
  # (m + 1). To get the number of examples with index -1 in front of
  # any example, we use the cumulative sum of the indicator vector
  # (indices_concat == -1).
  cum_sum = np.cumsum(indices_concat == -1)
  ranks = {k: cum_sum[indices_concat == i] + 1 for i, k in enumerate(keys)}
  exposures = {
      k: np.log2(len(list(perplexities_reference))) - np.log2(ranks[k])
      for k in ranks
  }
  return exposures


def compute_exposure_extrapolation(
    perplexities: Mapping[_KT, Sequence[float]],
    perplexities_reference: Sequence[float]) -> Mapping[_KT, Sequence[float]]:
  """Gets exposure using extrapolation.

  Args:
    perplexities: a `Mapping` where the key is an identifier for the secrets
      set, e.g. number of secret repetitions, and the value is an iterable of
      perplexities.
    perplexities_reference: perplexities of the random sequences that did not
      appear in the training data.

  Returns:
    The exposure of every secret measured using extrapolation, keyed in the same
    way as perplexities.
  """
  logging.info(
      'Will compute exposure (with extrapolation) for '
      'splits %s with %s examples using %s references.',
      str(perplexities.keys()), str([len(p) for p in perplexities.values()]),
      len(perplexities_reference))

  # Fit a skew normal distribution using the perplexities of the references
  snormal_param = stats.skewnorm.fit(perplexities_reference)

  # Estimate exposure using the fitted distribution
  exposures = {
      r: -np.log2(stats.skewnorm.cdf(p, *snormal_param))
      for r, p in perplexities.items()
  }
  return exposures

# Copyright 2021, The TensorFlow Authors.
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
"""Standard DpEvent classes."""

from typing import List

import attr


class DpEvent(object):
  """Base class for `DpEvent`s.

  A `DpEvent` describes a differentially private mechanism sufficiently for
  computing the associated privacy losses, both in isolation and in combination
  with other `DpEvent`s.
  """


@attr.s(frozen=True)
class NoOpDpEvent(DpEvent):
  """A `DpEvent` to represent operations with no privacy impact.

  A `NoOpDpEvent` is generally never required, but it can be useful as a
  placeholder where a `DpEvent` is expected, such as in tests or some live
  accounting pipelines.
  """


@attr.s(frozen=True)
class NonPrivateDpEvent(DpEvent):
  """A `DpEvent` to represent non-private operations.

  This `DpEvent` should be used when an operation is performed that does not
  satisfy (epsilon, delta)-DP. All `PrivacyAccountant`s should return infinite
  epsilon/delta when encountering a `NonPrivateDpEvent`.
  """


@attr.s(frozen=True)
class UnsupportedDpEvent(DpEvent):
  """A `DpEvent` to represent as-yet unsupported operations.

  This `DpEvent` should be used when an operation is performed that does not yet
  have any associated DP description, or if the description is temporarily
  inaccessible, for example, during development. All `PrivacyAccountant`s should
  return `is_supported(event)` is `False` for `UnsupportedDpEvent`.
  """


@attr.s(frozen=True, slots=True, auto_attribs=True)
class GaussianDpEvent(DpEvent):
  """The Gaussian mechanism."""
  noise_multiplier: float


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SelfComposedDpEvent(DpEvent):
  """A mechanism composed with itself multiple times."""
  event: DpEvent
  count: int


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ComposedDpEvent(DpEvent):
  """A series of composed mechanisms."""
  events: List[SelfComposedDpEvent]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class PoissonSampledDpEvent(DpEvent):
  """An application of Poisson subsampling."""
  sampling_probability: float
  event: DpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class FixedBatchSampledWrDpEvent(DpEvent):
  """Sampling exactly `batch_size` records with replacement."""
  dataset_size: int
  batch_size: int
  event: DpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class FixedBatchSampledWorDpEvent(DpEvent):
  """Sampling exactly `batch_size` records without replacement."""
  dataset_size: int
  batch_size: int
  event: DpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ShuffledDatasetDpEvent(DpEvent):
  """Shuffling a dataset and applying a mechanism to each partition."""
  partition_events: ComposedDpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class TreeAggregationDpEvent(DpEvent):
  """Applying a series of mechanisms with tree aggregation."""
  round_events: ComposedDpEvent
  max_record_occurences_across_all_rounds: int

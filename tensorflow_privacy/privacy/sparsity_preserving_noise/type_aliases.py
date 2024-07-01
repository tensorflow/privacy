# Copyright 2024, The TensorFlow Authors.
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
"""Type aliases for sparsity preserving noise."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import tensorflow as tf

InputArgs = Sequence[Any]
InputKwargs = Mapping[str, Any]
SparseGradient = tf.IndexedSlices
ContributionCountHistogram = tf.SparseTensor
ContributionCountHistogramFn = Callable[
    [SparseGradient], Mapping[str, ContributionCountHistogram]
]
NumMicrobatches = int | tf.Tensor
SparsityPreservingNoiseLayerRegistryFunction = Callable[
    [tf.keras.layers.Layer, InputArgs, InputKwargs, NumMicrobatches | None],
    ContributionCountHistogramFn,
]

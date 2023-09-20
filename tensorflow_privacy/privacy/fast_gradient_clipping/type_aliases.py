# Copyright 2023, The TensorFlow Authors.
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
"""A collection of type aliases used throughout the clipping library."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, Union
import tensorflow as tf


# Tensorflow aliases.
PackedTensors = Union[tf.Tensor, Iterable[tf.Tensor], Mapping[str, tf.Tensor]]

InputTensors = PackedTensors

OutputTensors = Union[tf.Tensor, Iterable[tf.Tensor]]

BatchSize = Union[int, tf.Tensor]

LossFn = Callable[..., tf.Tensor]

# Layer Registry aliases.
SquareNormFunction = Callable[[OutputTensors], tf.Tensor]

RegistryFunctionOutput = tuple[Any, OutputTensors, SquareNormFunction]

RegistryFunction = Callable[
    [
        Any,
        tuple[Any, ...],
        Mapping[str, Any],
        tf.GradientTape,
        Union[tf.Tensor, None],
    ],
    RegistryFunctionOutput,
]

# Clipping aliases.
GeneratorFunction = Optional[Callable[[Any, tuple, Mapping], tuple[Any, Any]]]

# Testing aliases.
LayerGenerator = Callable[[int, int], tf.keras.layers.Layer]

ModelGenerator = Callable[
    [LayerGenerator, Sequence[int], Sequence[int]], tf.keras.Model
]

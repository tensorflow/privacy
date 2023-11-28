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
"""Fast clipping function for `tf.keras.layers.Dense`."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases
from tensorflow_privacy.privacy.fast_gradient_clipping.registry_functions import einsum_utils


def dense_layer_computation(
    layer_instance: tf.keras.layers.Dense,
    input_args: Sequence[Any],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Optional[tf.Tensor] = None,
) -> type_aliases.RegistryFunctionOutput:
  """Registry function for `tf.keras.layers.Dense`.

  The logic for this computation is based on the following paper:
    https://arxiv.org/abs/1510.01799

  For the sake of efficiency, we fuse the variables and square grad norms
  for the kernel weights and bias vector together.

  Args:
    layer_instance: A `tf.keras.layers.Dense` instance.
    input_args: A `tuple` containing the first part of `layer_instance` input.
      Specifically, `layer_instance(*inputs_args, **input_kwargs)` should return
      a valid output.
    input_kwargs: A `tuple` containing the second part of `layer_instance`
      input. Specifically, `layer_instance(*inputs_args, **input_kwargs)` should
      return a valid output.
    tape: A `tf.GradientTape` instance that will be used to watch the output
      `base_vars`.
    num_microbatches: An optional numeric value or scalar `tf.Tensor` for
      indicating whether and how the losses are grouped into microbatches. If
      not None, num_microbatches must divide the batch size.

  Returns:
    A `tuple` `(base_vars, outputs, sqr_norm_fn)`. `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick,
    `outputs` is the result of `layer_instance(*inputs)`, and `sqr_norm_fn` is
    a function that takes one input, a `tf.Tensor` that represents the output
    of the call `tape.gradient(summed_loss, base_vars)` where `tape` is a
    `tf.GradientTape` instance that records the dense layer computation and
    `summed_loss` is the sum of the per-example losses of the underlying model.
    This function then returns the per-example squared L2 gradient norms of the
    trainable variables in `layer_instance`. These squared norms should be a 1D
    `tf.Tensor` of length `batch_size`.
  """
  if input_kwargs:
    raise ValueError("Dense layer calls should not receive kwargs.")
  del input_kwargs  # Unused in dense layer calls.
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*input_args)
  tape.watch(base_vars)
  layer_instance.activation = orig_activation
  outputs = orig_activation(base_vars) if orig_activation else base_vars

  def sqr_norm_fn(base_vars_grads):
    return einsum_utils.compute_fast_einsum_squared_gradient_norm(
        "...b,bc->...c",
        input_args[0],
        base_vars_grads,
        "c" if layer_instance.use_bias else None,
        num_microbatches,
    )

  return base_vars, outputs, sqr_norm_fn

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

from typing import Any, Mapping, Tuple, Union
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import common_manip_utils
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases


def dense_layer_computation(
    layer_instance: tf.keras.layers.Dense,
    input_args: Tuple[Any, ...],
    input_kwargs: Mapping[str, Any],
    tape: tf.GradientTape,
    num_microbatches: Union[tf.Tensor, None] = None,
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
    def _compute_gramian(x):
      if num_microbatches is not None:
        x_microbatched = common_manip_utils.maybe_add_microbatch_axis(
            x,
            num_microbatches,
        )
        return tf.matmul(x_microbatched, x_microbatched, transpose_b=True)
      else:
        # Special handling for better efficiency
        return tf.reduce_sum(tf.square(x), axis=tf.range(1, tf.rank(x)))

    inputs_gram = _compute_gramian(*input_args)
    base_vars_grads_gram = _compute_gramian(base_vars_grads)
    if layer_instance.use_bias:
      # Adding a bias term is equivalent to a layer with no bias term and which
      # adds an additional variable to the layer input that only takes a
      # constant value of 1.0. This is thus equivalent to adding 1.0 to the sum
      # of the squared values of the inputs.
      inputs_gram += 1.0
    return tf.reduce_sum(
        inputs_gram * base_vars_grads_gram,
        axis=tf.range(1, tf.rank(inputs_gram)),
    )

  return base_vars, outputs, sqr_norm_fn

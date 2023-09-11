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
"""Various helper functions related to `tf.keras.layers.EinsumDense`."""

import enum
import itertools
import re

import numpy as np
import tensorflow as tf

EquationType = enum.Enum(
    "EquationType",
    ["UNKNOWN", "NO_ELLIPSES", "LEFT_ELLIPSES", "RIGHT_ELLIPSES"],
)


def _is_batch_of_vectors(t: tf.Tensor) -> bool:
  """Checks if an input is a batch of (effectively) 1D vectors."""
  num_nontrivial_indices = 0
  for s in t.shape[1:]:
    if s > 1:
      num_nontrivial_indices += 1
    if num_nontrivial_indices > 1:
      return False
  return num_nontrivial_indices <= 1


def _parse_einsum_equation(
    equation: str,
) -> tuple[EquationType, tuple[str, str, str]]:
  """Returns the EquationType and I/O substrings of an einsum equation.

  Args:
    equation: The einsum equation `string`.

  Returns:
    A nested tuple `(equation_type, (ab_str, bc_str, ac_str))`, where
    `equation_type` specifies the type of einsum equation and `**_str`
    are the components of the equation. Excluding ellipses, the input equation
    should be of the form `ab,bc->ac` where `a`, `b`, and `c` can be themselves
    be substrings.

  Raises:
    ValueError: If `equation` is not a valid einsum equation in the context of
      the `tf.keras.layers.EinsumDense` layer.
  """

  def _try_match(regex_str):
    maybe_match = re.fullmatch(regex_str, equation)
    return maybe_match.groups() if maybe_match is not None else None

  groups1 = _try_match(r"([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)")
  if groups1 is not None:
    return EquationType.NO_ELLIPSES, groups1
  groups2 = _try_match(r"\.\.\.([a-zA-Z]+),([a-zA-Z]+)->\.\.\.([a-zA-Z]+)")
  if groups2 is not None:
    return EquationType.LEFT_ELLIPSES, groups2
  groups3 = _try_match(r"([a-zA-Z]+)\.\.\.,([a-zA-Z]+)->([a-zA-Z]+)\.\.\.")
  if groups3 is not None:
    return EquationType.RIGHT_ELLIPSES, groups3
  raise ValueError(
      "Invalid Einsum equation string "
      + equation
      + " ."
      "Must be one of the forms {ab,bc->ac}, {...ab,bc->...ac}, "
      "{ab...,bc->ac...}"
  )


def _reshape_einsum_inputs(
    input_tensor: tf.Tensor,
    equation: str,
) -> tf.Tensor:
  """Converts an input tensor of arbitrary rank to a batched matrix tensor.

  Args:
    input_tensor: A `tf.Tensor` corresponding to the first input of the einsum
      equation.
    equation: The einsum equation `string`.

  Returns:
    A rank-3 `tf.Tensor` representing a batch of rank-2 matrices with the same
    number of rows and columns. The output dimensions, in order, are:
    ```
    (num_batches, num_rows, num_columns)
    ```
    When `input_tensor` is a rank-2 `tf.Tensor`, the number of output rows is 1
    and the number of output columns is the second dimension of the input. The
    product of the non-trivial dimensions of the output should be equal to
    the product of the dimensions of `input_tensor`.
  """
  # Find the components `ab`, `bc`, and `ac` given that `equation` can only be
  # one of the following mutually exclusive forms:
  #
  #   (C1) ab,bc->ac,
  #   (C2) ...ab,bc->...ac
  #   (C3) ab...,bc->ac...
  #
  # NOTE: `a`, `b`, and `c` are (possibly) also substrings.

  # Compute the first index of the `b` part of the `ab` component.
  input_shape = input_tensor.shape
  input_len = len(input_shape)
  equation_type, (ab_str, bc_str, ac_str) = _parse_einsum_equation(equation)
  if equation_type == EquationType.LEFT_ELLIPSES:
    # In case (C2), the `a` part of this component can be empty, so we have no
    # choice but to compare the `c` part of `ac` with the `bc` component.
    c_len = 0
    for s1, s2 in itertools.zip_longest(reversed(bc_str), reversed(ac_str)):
      if s1 == s2:
        c_len += 1
      else:
        break
    b_len = len(bc_str) - c_len
    b_idx = input_len - b_len
  else:
    # For the other cases, we simply compare `ab` with `ac` to get the length
    # of the `a` component, i.e., the first index of `b`.
    b_idx = 0
    for s1, s2 in itertools.zip_longest(ab_str, ac_str):
      if s1 == s2:
        b_idx += 1
      else:
        break
  # Prepare `input_tensor` for reshaping and get the pivot index of the prepped
  # tensor. Note that case (C3) requires a transpose to ensure that matrix
  # multiplication is performed by the caller.
  if equation_type == EquationType.RIGHT_ELLIPSES:
    ellipses_idx = len(ab_str)
    # Convert `ab...` to `a...b`.
    new_ordering = (
        list(range(0, b_idx))
        + list(range(ellipses_idx, input_len))
        + list(range(b_idx, ellipses_idx))
    )
    input_tensor = tf.transpose(input_tensor, perm=new_ordering)
    ellipses_len = input_len - ellipses_idx
    pivot_idx = b_idx + ellipses_len
  else:
    pivot_idx = b_idx
  # The output tensor is a batched set of matrices, split at the pivot index
  # of the previously prepped tensor.
  base_tensor_shape = input_tensor.shape
  batch_size = base_tensor_shape[0]
  num_rows = int(np.prod(base_tensor_shape[1:pivot_idx]))
  num_columns = int(np.prod(base_tensor_shape[pivot_idx:]))
  return tf.reshape(input_tensor, shape=[batch_size, num_rows, num_columns])


def _reshape_einsum_outputs(
    output_tensor: tf.Tensor,
    equation: str,
) -> tf.Tensor:
  """Converts an output tensor of arbitrary rank to a batched matrix tensor.

  The logic is almost the same as in `_reshape_einsum_inputs()` except
  in the case where the equation is left-elided by ellipses. For this case,
  we need to pass in a reversed kernel shape.

  Args:
    output_tensor: A `tf.Tensor` corresponding to the output of the einsum
      equation.
    equation: The einsum equation `string`.

  Returns:
    A rank-3 `tf.Tensor` whose first dimension is the batch dimension. The
    product of the non-trivial dimensions of the output should be equal to
    the product of the non-trivial dimensions of `output_tensor`.
  """
  match = re.fullmatch(r"([a-zA-Z.]+),([a-zA-Z.]+)->([a-zA-Z.]+)", equation)
  if match is not None:
    s1, s2, s3 = match.groups()
  else:
    raise ValueError(
        "Invalid Einsum equation string "
        + equation
        + " ."
        "Must be one of the forms {ab,bc->ac}, {...ab,bc->...ac}, "
        "{ab...,bc->ac...}"
    )
  reversed_equation = s3 + "," + s2[::-1] + "->" + s1
  return _reshape_einsum_inputs(output_tensor, reversed_equation)

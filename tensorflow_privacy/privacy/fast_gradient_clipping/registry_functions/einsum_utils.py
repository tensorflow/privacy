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
import os
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


def _is_valid_einsum_equation(
    maybe_ab: str,
    maybe_bc: str,
    maybe_ac: str,
) -> bool:
  """Checks if three input strings form a valid einsum dense equation.

  Given three substrings `maybe_ab`, `maybe_bc`, and `maybe_ac`, this function
  checks if
  ```
  maybe_ab + ',' + maybe_bc + '->' + maybe_ac
  ```
  is an einsum equation of the form `ab,bc->ac`.

  Args:
    maybe_ab: The proposed `ab` substring.
    maybe_bc: The proposed `bc` substring.
    maybe_ac: The proposed `ac` substring.

  Returns:
    `True` if the three input strings form an einsum equation of the form
    `ab,bc->ac` and `False` otherwise.
  """
  a_substr = os.path.commonprefix([maybe_ab, maybe_ac])
  a_len = len(a_substr)
  b_substr = maybe_ab[a_len:]
  c_substr = maybe_ac[a_len:]
  return maybe_bc == b_substr + c_substr


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

  error_message = (
      "Invalid Einsum equation string "
      + equation
      + " ."
      "Must be one of the forms {ab,bc->ac}, {...ab,bc->...ac}, "
      "{ab...,bc->ac...}"
  )
  case_pairs = [
      # equation_type, regex_str
      (EquationType.NO_ELLIPSES, r"([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)"),
      (
          EquationType.LEFT_ELLIPSES,
          r"\.\.\.([a-zA-Z]+),([a-zA-Z]+)->\.\.\.([a-zA-Z]+)",
      ),
      (
          EquationType.RIGHT_ELLIPSES,
          r"([a-zA-Z]+)\.\.\.,([a-zA-Z]+)->([a-zA-Z]+)\.\.\.",
      ),
  ]
  for equation_type, regex_str in case_pairs:
    groups = _try_match(regex_str)
    if groups is not None:
      if not _is_valid_einsum_equation(*groups):
        raise ValueError(error_message)
      return equation_type, groups
  # No valid cases found. Raise an error.
  raise ValueError(error_message)


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

  Raises:
    ValueError: If `equation` is not a valid einsum equation in the context of
      the `tf.keras.layers.EinsumDense` layer.
  """
  # Find the components `ab`, `bc`, and `ac` given that `equation` can only be
  # one of the following mutually exclusive forms:
  #
  #   C1. ab,bc->ac,
  #   C2. ...ab,bc->...ac
  #   C3. ab...,bc->ac...
  #
  # NOTE: `a`, `b`, and `c` are (possibly) also substrings.

  # Compute the first index of the `b` part of the `ab` component.
  input_shape = input_tensor.shape
  input_len = len(input_shape)
  equation_type, (ab_str, bc_str, ac_str) = _parse_einsum_equation(equation)
  if equation_type == EquationType.LEFT_ELLIPSES:
    # In case C2, the `a` part of this component can be empty, so we have no
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
  # tensor. Note that case C3 requires a transpose to ensure that matrix
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

  Raises:
    ValueError: If `equation` is not a valid einsum equation in the context of
      the `tf.keras.layers.EinsumDense` layer.
  """
  # Get the raw components of the reversed equation.
  equation_type, (ab_str, bc_str, ac_str) = _parse_einsum_equation(equation)
  prefix = "..." if equation_type == EquationType.LEFT_ELLIPSES else ""
  suffix = "..." if equation_type == EquationType.RIGHT_ELLIPSES else ""
  ellided_ab_str = prefix + ab_str + suffix
  ellided_ac_str = prefix + ac_str + suffix
  # Swap the `b` and `c` components.
  c_str = os.path.commonprefix([bc_str[::-1], ac_str[::-1]])[::-1]
  b_len = len(bc_str) - len(c_str)
  b_str = bc_str[:b_len]
  cb_str = c_str + b_str
  reversed_equation = ellided_ac_str + "," + cb_str + "->" + ellided_ab_str
  return _reshape_einsum_inputs(output_tensor, reversed_equation)


def _get_einsum_bias_adjoint_reduction_axes(
    equation: str,
    bias_axes: str,
    einsum_rank: int,
) -> list[int]:
  """Computes axes related to the per-example adjoint of the einsum bias op.

  To describe the output of this computation, first recall that for each
  example the `EinsumDense` layer performs the following transformation:
  ```
  F(W, bias | X) = Einsum(W, X) + Q(bias)
  ```
  where `W` is a tensor of trainable variables, `bias` is a tensor of rank
  `len(bias_axes)`, `X` is a batch of inputs, and `Q` is a linear broadcast
  operator that roughly corresponds to `Q(bias) ~= tf.broadcast_to(bias, S)` for
  `S := tf.shape(Einsum(W, X))`.

  It is straightforward to show that the per-example adjoint of `Q` is given by
  `Q'(Y) := tf.reduce_sum(Y, axes=R)` where `R` contains the broadcasting
  indices. This function returns `R` as an unordered list of `int`s.

  Assumptions:

    A1. `equation` is one of the following forms:
          C1. `ab,bc->ac`
          C2. `...ab,bc->...ac`
          C3. `ab...,bc->ac...`

    A2. The first character in the substring `a` (or `...a` in C2)
        in assumption A1 corresponds to the batch dimension.

    A3. The characters in `bias_axes` must be subset of the non-batch dimension
        characters in the substring `ac` (or `...ac` in C2) in
        assumption A1.

    A4. `einsum_rank` is the length of the substring `ac` (or `...ac` in C2) in
        assumption A1. This includes the batch dimension.

  Examples:

    1. equation = 'ab,bc->ac', bias_axes = 'c', einsum_rank = 2 -> []
    2. equation = 'ab,bce->ace', bias_axes = 'ce', einsum_rank = 3, -> []
    3. equation = 'ab,bce->ace', bias_axes = 'c', einsum_rank = 3, -> [2]
    4. equation = 'ab,bce->ace', bias_axes = 'e', einsum_rank = 3, -> [1]
    5. equation = 'ab,bced->aced', bias_axes = 'ced', einsum_rank = 4 -> []
    6. equation = 'ab,bced->aced', bias_axes = 'ce', einsum_rank = 4, -> [3],
    7. equation = 'ab,bced->aced', bias_axes = 'c', einsum_rank = 4, -> [2, 3]
    8. equation = '...ab,bce->...ace', bias_axes = 'c', einsum_rank = 4
       -> [1, 3]
    9. equation = '...ab,bce->...ace', bias_axes = 'c', einsum_rank = 10
       -> [1, 2, 3, 4, 5, 6, 7, 9]
    10. equation = 'ab...,bce->ace...', bias_axes = 'e', einsum_rank = 4
       -> [1, 3]

  Args:
    equation: The einsum equation `string`.
    bias_axes: A substring of the output part of `equation` specifying which
      axes a bias `tf.Tensor` is added to.
    einsum_rank: The rank of the tensor that the per-example adjoint operator is
      being applied to.

  Returns:
    A list of `int` containing axes in the `input` corresponding to
    `input_rank`. Each `int` is at most `input_rank-1` and excludes zero.

  Raises:
    ValueError: If `equation` is not a valid einsum equation in the context of
      the `tf.keras.layers.EinsumDense` layer.
  """
  reduction_axes = []
  bias_char_set = set(bias_axes)
  equation_type, (_, _, ac_str) = _parse_einsum_equation(equation)
  # Do not allow the bias axes to be the batch axis, since we want the adjoint
  # of the bias broadcast op to apply the same operation to all examples in a
  # batch.
  if equation_type != EquationType.LEFT_ELLIPSES and ac_str[0] in bias_axes:
    raise ValueError(f"Bias axis '{bias_axes}' cannot also be the batch axis.")
  # If `equation` of the form `...ab,bc->...ac`, i.e., case C2, we do a
  # right to left traversal; the other cases do a left to right traversal.
  input_indices = range(einsum_rank)
  traversal_zip = (
      itertools.zip_longest(reversed(input_indices), reversed(ac_str))
      if equation_type == EquationType.LEFT_ELLIPSES
      else itertools.zip_longest(input_indices, ac_str)
  )
  # Traverse the output part of `equation` and add an index to the output if
  # the corresponding `char` in the `ac` part is NOT in `bias_axes` and the
  # index is not zero (batch dimension). Add all indices except index zero in
  # the `...` part of the output substring (if present).
  for idx, output_char in traversal_zip:
    # Exclude the batch dimension (idx == 0), since we want the per-example
    # adjoint.
    if idx != 0:
      if output_char is not None and bias_char_set:
        if output_char not in bias_char_set:
          reduction_axes.append(idx)
        else:
          bias_char_set.remove(output_char)
      else:
        reduction_axes.append(idx)
  return reduction_axes

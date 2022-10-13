# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per example gradients clipping and aggregation for sparse gradients.

Modified from tape.jacobian to support sparse gradients.
"""
import sys
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import six
import tensorflow as tf

from tensorflow.python.ops.parallel_for import control_flow_ops  # pylint: disable=g-direct-tensorflow-import

GradientTensor = Union[tf.Tensor, tf.IndexedSlices]
T = TypeVar('T')
Nested = Union[T, Tuple[Any, ...], List[Any], Dict[str, Any]]


def _deduplicate_batch_indexed_slices(
    batched_values: tf.Tensor,
    indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Removes duplication of indexed slices by summing them."""
  perm = tf.concat([
      tf.constant([1, 0], dtype=tf.int32),
      tf.range(tf.rank(batched_values))[2:]
  ],
                   axis=0)
  batched_values = tf.transpose(batched_values, perm=perm)
  unique_indices, new_pos = tf.unique(indices)
  summed_values = tf.math.unsorted_segment_sum(batched_values, new_pos,
                                               tf.shape(unique_indices)[0])
  return tf.transpose(summed_values, perm=perm), unique_indices


def _batch_global_norm(vals: List[tf.Tensor]) -> tf.Tensor:
  """Computes the global norm for each row in the batch."""

  def _norm_squared(v):
    return tf.cast(
        tf.reduce_sum(
            tf.reshape(tf.square(v), tf.stack([tf.shape(v)[0], -1])), axis=1),
        tf.float32)

  return tf.sqrt(tf.add_n([_norm_squared(v) for v in vals if v is not None]))


def _batch_clip_by_global_norm(
    vals: List[tf.Tensor], normalize: bool,
    l2_norm_clip: Optional[float]) -> List[tf.Tensor]:
  """Batch clips by global norm with normalize option."""
  batch_global_norm = _batch_global_norm(vals)
  if l2_norm_clip is None:
    l2_norm_clip = 1.0
  clip_ratio = l2_norm_clip / tf.maximum(batch_global_norm, 1e-8)
  if not normalize:
    clip_ratio = tf.minimum(1.0, clip_ratio)

  def _expand_dims(e, v):
    new_shape = tf.concat(
        [tf.shape(v)[0:1],
         tf.ones_like(tf.shape(v), dtype=tf.int32)[:-1]],
        axis=0)
    return tf.reshape(e, new_shape)

  return [
      v *
      _expand_dims(tf.cast(clip_ratio, v.dtype), v) if v is not None else None
      for v in vals
  ]


def clip_and_aggregate_gradients(
    tape: tf.GradientTape,
    target: tf.Tensor,
    sources: Nested[tf.Tensor],
    unconnected_gradients: tf.UnconnectedGradients = tf.UnconnectedGradients
    .NONE,
    normalize: bool = False,
    l2_norm_clip: Optional[float] = None,
    aggregate_method: str = 'mean',
    keep_sparse_threshold: int = 10000) -> Nested[GradientTensor]:
  """Clips (per-example) and aggregates gradients.

  This procedure computes the Jacobian with respect to a vectorized loss,
  i.e. the `target` argument, clips the gradient with repsect to each
  individual output, and sums the clipped gradients. This is correct as
  per-example gradient if there is a one to one mapping from the input example
  to the output loss.

  Args:
    tape: a persistent tape.
    target: Tensor to be differentiated. It is assumed that each value in
      `target` is associated with an example so the gradient clipping would be
      applied to the vectorized target.
    sources: a list or nested structure of Tensors or Variables. `target` will
      be differentiated against elements in `sources`.
    unconnected_gradients: a value which can either hold 'none' or 'zero' and
      alters the value which will be returned if the `target` and `sources` are
      unconnected. The possible values and effects are detailed in
      'UnconnectedGradients' and it defaults to 'none'.
    normalize: whether to normalize each gradient.
    l2_norm_clip: when `normalize` is `True`, every gradient is scaled to
      `l2_norm_clip` (which can be set to None, understood as 1). When
      `normalize` is `False`, it performs the regular clipping, i.e. scaling the
      gradient to `l2_norm_clip` only if the gradient's L2 norm is larger than
      `l2_norm_clip`. When `l2_norm_clip` is `None`, do nothing.
    aggregate_method: the method for aggregating the gradients. Currently only
      supports `sum` and `mean`, default to `mean`.
    keep_sparse_threshold: when the gradient is a `tf.IndexedSlices`,
      `keep_sparse_threshold` is used to determine if we should keep it in its
      sparse representation (when the number of embedding items, i.e. vocabulary
      size >= `keep_sparse_threshold`) or convert it into a dense tensor (when <
      `keep_sparse_threshold`). The reason for this parameter is that the
      current implementation of embedding lookup merges all the indices in a
      batch, hence the sparse representation has input size the same as the
      number of indices. When it is larger than the embedding size, it would be
      more efficient to convert the sparse representation to a dense tensor. So
      this threshold should be set around the number of indices in a typical
      batch. When it is -1, always convert the sparse tensor to a dense tensor.

  Returns:
    Gradients stored in the same structure as `sources` with a one to one
    mapping to the variables in `sources`. Each gradients may be a dense
    tensor or a `tf.IndexedSlices`.

  Raises:
    RuntimeError: if `tape` is not persistent.
    ValueError: if aggregate_method is not 'mean' or 'sum'.
  """

  if tape._tape is None:  # pylint: disable=protected-access
    raise RuntimeError('A non-persistent GradientTape can only be used to '
                       'compute one set of gradients (or jacobians)')

  if aggregate_method not in ['mean', 'sum']:
    raise ValueError('Only mean and sum methods are supported. But got '
                     f'{aggregate_method}')

  flat_sources = tf.nest.flatten(sources)
  # Note that we push and pop the tape here and below. This is needed since we
  # need gradients through the enclosed operations.
  with tape._ensure_recording():  # pylint: disable=protected-access
    target = tf.reshape(target, [-1])
  target_shape = target.shape

  convert_to_dense_indicator = [True for _ in flat_sources]
  if keep_sparse_threshold >= 0:
    convert_to_dense_indicator = [
        s.shape[0] < keep_sparse_threshold for s in flat_sources
    ]

  def _unpack_indexed_slices(x, convert_to_dense):
    """Optionally unpacks `tf.IndexedSlices` to dict of three dense tensors."""
    if convert_to_dense or not isinstance(x, tf.IndexedSlices):
      # If x is kept as a tf.IndexedSlices, it will be converted to a dense
      # tensor in pfor.
      return x
    return {
        'indices': x.indices,
        'values': x.values,
        'dense_shape': x.dense_shape
    }

  def loop_fn(i):
    with tape._ensure_recording():  # pylint: disable=protected-access
      y = tf.gather(target, i)
    g = tape.gradient(
        y, flat_sources, unconnected_gradients=unconnected_gradients)
    g = tf.nest.map_structure(_unpack_indexed_slices, g,
                              convert_to_dense_indicator)
    return g

  try:
    target_size = int(target.shape[0])
  except TypeError:
    # When the shape is unavailable, fall back to the tensor op.
    target_size = tf.shape(target)[0]

  try:
    output = control_flow_ops.pfor(loop_fn, target_size)
  except ValueError as err:
    six.reraise(
        ValueError,
        ValueError(
            str(err) + '\nEncountered an exception while vectorizing the '
            'jacobian computation. Consider using a non-vectorized version, '
            'i.e. by computing the gradient for each output sequentially.'),
        sys.exc_info()[2])

  grads = []
  for i, out in enumerate(output):
    if out is not None:
      # Determines if the output is a unpacked tf.IndexedSlices. Since `sources`
      # has been flattened, it is only when the output is a dictionary (of three
      # dense tensors).
      if not isinstance(out, dict):
        if tf.executing_eagerly():
          out.set_shape(target_shape.concatenate(flat_sources[i].shape))
        grads.append((out, None, None))
      else:
        # Remove duplicates at per-example level. This is for both correctness
        # (when the same index gets gathered more than once in the same example)
        # and efficiency (for the subsequent clipping). All the examples in
        # the batch should have the same indices so it suffices to take the
        # first row.
        values, indices = _deduplicate_batch_indexed_slices(
            out['values'], out['indices'][0])
        # The `dense_shape` of all the examples are the same so we take the
        # first row.
        grads.append((values, indices, out['dense_shape'][0]))
    else:
      grads.append((None, None, None))

  if normalize or l2_norm_clip is not None:
    values, indices, dense_shape = zip(*grads)
    values = _batch_clip_by_global_norm(values, normalize, l2_norm_clip)
    grads = zip(values, indices, dense_shape)

  new_output = []
  for values, indices, dense_shape in grads:
    if values is None:
      new_output.append(None)
      continue
    if aggregate_method == 'sum':
      values = tf.reduce_sum(values, axis=0)
    else:
      values = tf.reduce_mean(values, axis=0)
    if indices is None:
      new_output.append(values)
    else:
      new_output.append(
          tf.IndexedSlices(
              values=values, indices=indices, dense_shape=dense_shape))
  return tf.nest.pack_sequence_as(sources, new_output)

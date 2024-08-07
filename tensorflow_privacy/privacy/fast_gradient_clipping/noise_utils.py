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
"""Utility functions that help in adding noise to gradients."""

from collections.abc import Sequence
from typing import Literal, Optional

from absl import logging
import tensorflow as tf


def _infer_loss_reduction_type(model: tf.keras.Model):
  """Infers what type of loss reduction is being performed."""
  model_loss = model.loss
  if isinstance(model_loss, tf.keras.losses.Loss):
    return model_loss.reduction
  elif isinstance(model.loss, dict):
    reductions = set()
    compiled_loss = model.compiled_loss
    if compiled_loss is None:
      raise ValueError('Model must be compiled for adding noise')
    new_config_list = compiled_loss.get_config()['losses']
    for loss_config in new_config_list:
      reductions.add(loss_config['config']['reduction'])
    if len(reductions) > 1:
      raise ValueError(
          'Reductions in models with multiple losses must all be the same'
      )
    return reductions.pop()
  else:
    raise ValueError(
        'Unsupported type for adding noise: {}'.format(type(model_loss))
    )


def add_aggregate_noise(
    clipped_grads: list[tf.Tensor],
    batch_size: tf.Tensor,
    l2_norm_clip: float,
    noise_multiplier: float,
    loss_reduction: Optional[Literal['mean', 'sum']] = None,
    loss_model: Optional[tf.keras.Model] = None,
) -> Sequence[tf.Tensor]:
  """Adds noise to a collection of clipped gradients.

  The magnitude of the noise depends on the aggregation strategy of the
  input model's loss function.

  Args:
    clipped_grads: A list of `tf.Tensor`s representing the clipped gradients.
    batch_size: The batch size. Used for normalizing the noise when
      `loss_reduction` is 'sum'.
    l2_norm_clip: Clipping norm (max L2 norm of each gradient).
    noise_multiplier: Ratio of the standard deviation to the clipping norm.
    loss_reduction: An string description of how the loss is reduced over
      examples. Currently supports 'mean' and 'sum'. If `None`, then the
      aggregation type must be inferred from `input_model.loss`.
    loss_model: An optional `tf.keras.Model` used to infer the loss reduction
      strategy from if `loss_reduction` is `None`.

  Returns:
    A list of tensors containing the clipped gradients, but with the right
    amount of Gaussian noise added to them (depending on the reduction
    strategy of the loss function).

  Raises:
    ValueError: If both `loss_model` and `loss_reduction` are `None` or if
      they are both not `None`.
  """
  if loss_reduction is None and loss_model is None:
    raise ValueError(
        'Exactly one of `loss_reduction` and `loss_model` must be populated.'
        ' Instead, both arguments were `None`.'
    )
  if loss_reduction is not None and loss_model is not None:
    raise ValueError(
        'Exactly one of `loss_reduction` and `loss_model` must be populated.'
        ' Instead, both arguments were not `None`.'
    )

  if loss_reduction is None and loss_model is not None:
    implicit_mean_reductions = [
        tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        tf.keras.losses.Reduction.AUTO,
    ]
    model_reduction = _infer_loss_reduction_type(loss_model)
    loss_reduction = (
        'mean' if model_reduction in implicit_mean_reductions else 'sum'
    )
    if model_reduction == tf.keras.losses.Reduction.AUTO:
      logging.info(
          'Assuming that the model loss reduction is `SUM_OVER_BATCH_SIZE`.'
      )

  scale = l2_norm_clip
  if loss_reduction == 'mean':
    scale /= tf.cast(batch_size, tf.float32)

  def add_noise(g):
    return g + tf.random.normal(
        tf.shape(g), mean=0.0, stddev=noise_multiplier * scale
    )

  return tf.nest.map_structure(add_noise, clipped_grads)

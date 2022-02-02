# Copyright 2020, The TensorFlow Authors.
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
"""Utility functions for writing attack results to tensorboard."""

from typing import List, Union

import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import get_flattened_attack_metrics


def write_to_tensorboard_tf2(writers, tags, values, step):
  """Write metrics to tensorboard.

  Args:
    writers: a list of tensorboard writers or one writer to be used for metrics.
      If it's a list, it should be of the same length as tags
    tags: a list of tags of metrics
    values: a list of values of metrics with the same length as tags
    step: step for the tensorboard summary
  """
  if writers is None or not writers:
    raise ValueError('write_to_tensorboard does not get any writer.')

  if not isinstance(writers, list):
    writers = [writers] * len(tags)

  assert len(writers) == len(tags) == len(values)

  for writer, tag, val in zip(writers, tags, values):
    with writer.as_default():
      tf.summary.scalar(tag, val, step=step)
      writer.flush()

  for writer in set(writers):
    with writer.as_default():
      writer.flush()


def write_results_to_tensorboard_tf2(
    attack_results: AttackResults,
    writers: Union[tf.summary.SummaryWriter, List[tf.summary.SummaryWriter]],
    step: int, merge_classifiers: bool):
  """Write attack results to tensorboard.

  Args:
    attack_results: results from attack
    writers: a list of tensorboard writers or one writer to be used for metrics
    step: step for the tensorboard summary
    merge_classifiers: if true, plot different classifiers with the same
      slicing_spec and metric in the same figure
  """
  if writers is None or not writers:
    raise ValueError('write_results_to_tensorboard does not get any writer.')

  att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
      attack_results)
  if merge_classifiers:
    att_tags = ['attack/' + f'{s}_{m}' for s, m in zip(att_slices, att_metrics)]
    write_to_tensorboard_tf2([writers[t] for t in att_types], att_tags,
                             att_values, step)
  else:
    att_tags = [
        'attack/' + f'{s}_{t}_{m}'
        for t, s, m in zip(att_types, att_slices, att_metrics)
    ]
    write_to_tensorboard_tf2(writers, att_tags, att_values, step)

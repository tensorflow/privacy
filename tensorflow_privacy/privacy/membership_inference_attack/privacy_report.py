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

# Lint as: python3
"""Plotting code for ML Privacy Reports."""
from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults


def plot_by_epochs(results: Iterable[AttackResults],
                   privacy_metrics: Iterable[str]) -> plt.Figure:
  """Plots privacy vulnerabilities by epochs."""
  _validate_results(results)
  all_results_df = None
  for attack_results in results:
    attack_results_df = attack_results.calculate_pd_dataframe()
    attack_results_df = attack_results_df.loc[attack_results_df['slice feature']
                                              == 'entire_dataset']
    attack_results_df.insert(0, 'Epoch',
                             attack_results.privacy_report_metadata.epoch_num)
    if all_results_df is None:
      all_results_df = attack_results_df
    else:
      all_results_df = pd.concat([all_results_df, attack_results_df],
                                 ignore_index=True)

  fig, axes = plt.subplots(1, len(privacy_metrics))
  if len(privacy_metrics) == 1:
    axes = (axes,)
  for i, privacy_metric in enumerate(privacy_metrics):
    attack_types = all_results_df['attack type'].unique()
    for attack_type in attack_types:
      axes[i].plot(
          all_results_df.loc[all_results_df['attack type'] == attack_type]
          ['Epoch'], all_results_df.loc[all_results_df['attack type'] ==
                                        attack_type][privacy_metric])
    axes[i].legend(attack_types)
    axes[i].set_xlabel('Epoch')
    axes[i].set_title('%s for Entire dataset' % privacy_metric)

  return fig


def _validate_results(results: Iterable[AttackResults]):
  for attack_results in results:
    if not attack_results or not attack_results.privacy_report_metadata:
      raise ValueError('Privacy metadata is not defined.')
    if not attack_results.privacy_report_metadata.epoch_num:
      raise ValueError('epoch_num in metadata is not defined.')

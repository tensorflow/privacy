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
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import PrivacyMetric


def plot_by_epochs(results: Iterable[AttackResults],
                   privacy_metrics: Iterable[PrivacyMetric]) -> plt.Figure:
  """Plots privacy vulnerabilities vs epoch numbers for a single model variant.

  In case multiple privacy metrics are specified, the plot will feature
  multiple subplots (one subplot per metrics).
  Args:
    results: AttackResults for the plot
    privacy_metrics: List of enumerated privacy metrics that should be plotted.

  Returns:
    A pyplot figure with privacy vs accuracy plots.
  """

  _validate_results(results)
  all_results_df = _calculate_combined_df_with_metadata(results)
  return _generate_subplots(
      all_results_df=all_results_df,
      x_axis_metric='Epoch',
      figure_title='Vulnerability per Epoch',
      privacy_metrics=privacy_metrics)


def plot_privacy_vs_accuracy_single_model(
    results: Iterable[AttackResults], privacy_metrics: Iterable[PrivacyMetric]):
  """Plots privacy vulnerabilities vs accuracy plots for a single model variant.

  In case multiple privacy metrics are specified, the plot will feature
  multiple subplots (one subplot per metrics).
  Args:
    results: AttackResults for the plot
    privacy_metrics: List of enumerated privacy metrics that should be plotted.

  Returns:
    A pyplot figure with privacy vs accuracy plots.

  """
  _validate_results(results)
  all_results_df = _calculate_combined_df_with_metadata(results)
  return _generate_subplots(
      all_results_df=all_results_df,
      x_axis_metric='Train accuracy',
      figure_title='Privacy vs Utility Analysis',
      privacy_metrics=privacy_metrics)


def _calculate_combined_df_with_metadata(results: Iterable[AttackResults]):
  """Adds metadata to the dataframe and concats them together."""
  all_results_df = None
  for attack_results in results:
    attack_results_df = attack_results.calculate_pd_dataframe()
    attack_results_df = attack_results_df.loc[attack_results_df['slice feature']
                                              == 'entire_dataset']
    attack_results_df.insert(0, 'Epoch',
                             attack_results.privacy_report_metadata.epoch_num)
    attack_results_df.insert(
        0, 'Train accuracy',
        attack_results.privacy_report_metadata.accuracy_train)
    attack_results_df.insert(
        0, 'legend label',
        attack_results.privacy_report_metadata.model_variant_label + ' - ' +
        attack_results_df['attack type'])
    if all_results_df is None:
      all_results_df = attack_results_df
    else:
      all_results_df = pd.concat([all_results_df, attack_results_df],
                                 ignore_index=True)
  return all_results_df


def _generate_subplots(all_results_df: pd.DataFrame, x_axis_metric: str,
                       figure_title: str,
                       privacy_metrics: Iterable[PrivacyMetric]):
  """Create one subplot per privacy metric for a specified x_axis_metric."""
  fig, axes = plt.subplots(1, len(privacy_metrics))
  # Set a title for the entire group of subplots.
  fig.suptitle(figure_title)
  if len(privacy_metrics) == 1:
    axes = (axes,)
  for i, privacy_metric in enumerate(privacy_metrics):
    legend_labels = all_results_df['legend label'].unique()
    for legend_label in legend_labels:
      single_label_results = all_results_df.loc[all_results_df['legend label']
                                                == legend_label]
      axes[i].plot(single_label_results[x_axis_metric],
                   single_label_results[str(privacy_metric)])
    axes[i].legend(legend_labels)
    axes[i].set_xlabel(x_axis_metric)
    axes[i].set_title('%s for Entire dataset' % str(privacy_metric))

  return fig


def _validate_results(results: Iterable[AttackResults]):
  for attack_results in results:
    if not attack_results or not attack_results.privacy_report_metadata:
      raise ValueError('Privacy metadata is not defined.')
    if not attack_results.privacy_report_metadata.epoch_num:
      raise ValueError('epoch_num in metadata is not defined.')

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
"""An example for the membership inference attacks.

This is using a toy model based on classifying four spacial clusters of data.
"""

import os
import tempfile

from absl import app
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn import metrics
import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import data_structures
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report


def generate_random_cluster(center, scale, num_points):
  return np.random.normal(size=(num_points, len(center))) * scale + center


def generate_features_and_labels(samples_per_cluster=250, scale=0.1):
  """Generates noised 3D clusters."""
  cluster_centers = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]

  features = np.concatenate((
      generate_random_cluster(
          center=cluster_centers[0],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[1],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[2],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[3],
          scale=scale,
          num_points=samples_per_cluster),
  ))

  # Cluster labels: 0, 1, 2 and 3
  labels = np.concatenate((
      np.zeros(samples_per_cluster),
      np.ones(samples_per_cluster),
      np.ones(samples_per_cluster) * 2,
      np.ones(samples_per_cluster) * 3,
  )).astype("uint8")

  return (features, labels)


def get_models(num_clusters):
  """Get the two models we will be using."""
  # Hint: play with the number of layers to achieve different level of
  # over-fitting and observe its effects on membership inference performance.
  three_layer_model = tf.keras.Sequential([
      tf.keras.layers.Dense(300, activation="relu"),
      tf.keras.layers.Dense(300, activation="relu"),
      tf.keras.layers.Dense(300, activation="relu"),
      tf.keras.layers.Dense(num_clusters, activation="relu"),
      tf.keras.layers.Softmax()
  ])
  three_layer_model.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"])

  two_layer_model = tf.keras.Sequential([
      tf.keras.layers.Dense(300, activation="relu"),
      tf.keras.layers.Dense(300, activation="relu"),
      tf.keras.layers.Dense(num_clusters, activation="relu"),
      tf.keras.layers.Softmax()
  ])
  two_layer_model.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"])
  return three_layer_model, two_layer_model


def main(unused_argv):
  # Hint: Play with "noise_scale" for different levels of overlap between
  # the generated clusters. More noise makes the classification harder.
  noise_scale = 2
  training_features, training_labels = generate_features_and_labels(
      samples_per_cluster=250, scale=noise_scale)
  test_features, test_labels = generate_features_and_labels(
      samples_per_cluster=250, scale=noise_scale)

  num_clusters = int(round(np.max(training_labels))) + 1

  three_layer_model, two_layer_model = get_models(num_clusters)
  models = {
      "two_layer_model": two_layer_model,
      "three_layer_model": three_layer_model,
  }

  num_epochs_per_round = 20
  epoch_results = data_structures.AttackResultsCollection([])
  for model_name, model in models.items():
    print(f"Train {model_name}.")
    # Incrementally train the model and store privacy metrics
    # every num_epochs_per_round.
    for i in range(5):
      model.fit(
          training_features,
          training_labels,
          validation_data=(test_features, test_labels),
          batch_size=64,
          epochs=num_epochs_per_round,
          shuffle=True)

      training_pred = model.predict(training_features)
      test_pred = model.predict(test_features)

      # Add metadata to generate a privacy report.
      privacy_report_metadata = data_structures.PrivacyReportMetadata(
          accuracy_train=metrics.accuracy_score(
              training_labels, np.argmax(training_pred, axis=1)),
          accuracy_test=metrics.accuracy_score(test_labels,
                                               np.argmax(test_pred, axis=1)),
          epoch_num=num_epochs_per_round * (i + 1),
          model_variant_label=model_name)

      attack_results = mia.run_attacks(
          data_structures.AttackInputData(
              labels_train=training_labels,
              labels_test=test_labels,
              probs_train=training_pred,
              probs_test=test_pred),
          data_structures.SlicingSpec(entire_dataset=True, by_class=True),
          attack_types=(data_structures.AttackType.THRESHOLD_ATTACK,
                        data_structures.AttackType.LOGISTIC_REGRESSION),
          privacy_report_metadata=privacy_report_metadata)
      epoch_results.append(attack_results)

  # Generate privacy reports
  epoch_figure = privacy_report.plot_by_epochs(epoch_results, [
      data_structures.PrivacyMetric.ATTACKER_ADVANTAGE,
      data_structures.PrivacyMetric.AUC
  ])
  epoch_figure.show()
  privacy_utility_figure = privacy_report.plot_privacy_vs_accuracy(
      epoch_results, [
          data_structures.PrivacyMetric.ATTACKER_ADVANTAGE,
          data_structures.PrivacyMetric.AUC
      ])
  privacy_utility_figure.show()

  # Example of saving the results to the file and loading them back.
  with tempfile.TemporaryDirectory() as tmpdirname:
    filepath = os.path.join(tmpdirname, "results.pickle")
    attack_results.save(filepath)
    loaded_results = data_structures.AttackResults.load(filepath)
    print(loaded_results.summary(by_slices=False))

  # Print attack metrics
  for attack_result in attack_results.single_attack_results:
    print("Slice: %s" % attack_result.slice_spec)
    print("Attack type: %s" % attack_result.attack_type)
    print("AUC: %.2f" % attack_result.roc_curve.get_auc())

    print("Attacker advantage: %.2f\n" %
          attack_result.roc_curve.get_attacker_advantage())

  max_auc_attacker = attack_results.get_result_with_max_auc()
  print("Attack type with max AUC: %s, AUC of %.2f" %
        (max_auc_attacker.attack_type, max_auc_attacker.roc_curve.get_auc()))

  max_advantage_attacker = attack_results.get_result_with_max_attacker_advantage(
  )
  print("Attack type with max advantage: %s, Attacker advantage of %.2f" %
        (max_advantage_attacker.attack_type,
         max_advantage_attacker.roc_curve.get_attacker_advantage()))

  # Print summary
  print("Summary without slices: \n")
  print(attack_results.summary(by_slices=False))

  print("Summary by slices: \n")
  print(attack_results.summary(by_slices=True))

  # Print pandas data frame
  print("Pandas frame: \n")
  pd.set_option("display.max_rows", None, "display.max_columns", None)
  print(attack_results.calculate_pd_dataframe())

  # Example of ROC curve plotting.
  figure = plotting.plot_roc_curve(
      attack_results.single_attack_results[0].roc_curve)
  figure.show()
  plt.show()

  # For saving a figure into a file:
  # plotting.save_plot(figure, <file_path>)

  # Let's look at the per-example membership scores. We'll look at how the
  # scores from threshold and logistic regression attackers correlate.

  # We take the MIA result of the final three layer model
  sample_model = epoch_results.attack_results_list[-1]
  print("We will look at the membership scores of",
        sample_model.privacy_report_metadata.model_variant_label, "at epoch",
        sample_model.privacy_report_metadata.epoch_num)
  sample_results = sample_model.single_attack_results

  # The first two entries of sample_results are from the threshold and
  # logistic regression attackers on the whole dataset.
  print("Correlation between the scores of the following two attackers:", "\n ",
        sample_results[0].slice_spec, sample_results[0].attack_type, "\n ",
        sample_results[1].slice_spec, sample_results[1].attack_type)
  threshold_results = np.concatenate(  # scores by threshold attacker
      (sample_results[0].membership_scores_train,
       sample_results[0].membership_scores_test))
  lr_results = np.concatenate(  # scores by logistic regression attacker
      (sample_results[1].membership_scores_train,
       sample_results[1].membership_scores_test))

  # Order the scores and plot them
  threshold_orders = scipy.stats.rankdata(threshold_results)
  lr_orders = scipy.stats.rankdata(lr_results)

  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  axes.scatter(threshold_orders, lr_orders, alpha=0.2, linewidths=0)
  m, b = np.polyfit(threshold_orders, lr_orders, 1)  # linear fit
  axes.plot(threshold_orders, m * threshold_orders + b, color="orange")
  axes.set_aspect("equal", adjustable="box")
  fig.show()


if __name__ == "__main__":
  app.run(main)

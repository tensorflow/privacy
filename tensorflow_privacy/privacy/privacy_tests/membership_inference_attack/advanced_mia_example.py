# Copyright 2022, The TensorFlow Authors.
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
"""An example for using advanced_mia."""

import functools
import gc
import os
from typing import Optional

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests import utils
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import plotting as mia_plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData

FLAGS = flags.FLAGS
_LR = flags.DEFINE_float('learning_rate', 0.02, 'Learning rate for training')
_BATCH = flags.DEFINE_integer('batch_size', 250, 'Batch size')
_EPOCHS = flags.DEFINE_integer('epochs', 20, 'Number of epochs')
_NUM_SHADOWS = flags.DEFINE_integer('num_shadows', 10,
                                    'Number of shadow models.')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'Model directory.')


def small_cnn():
  """Setup a small CNN for image classification."""
  model = tf.keras.models.Sequential()
  # Add a layer to do random horizontal augmentation.
  model.add(tf.keras.layers.RandomFlip('horizontal'))
  model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

  for _ in range(3):
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10))
  return model


def load_cifar10():
  """Loads CIFAR10, with training and test combined."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
  y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()
  return x, y


def plot_curve_with_area(x, y, xlabel, ylabel, ax, label, title=None):
  ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
  ax.plot(x, y, lw=2, label=label)
  ax.set(xlabel=xlabel, ylabel=ylabel)
  ax.set(aspect=1, xscale='log', yscale='log')
  ax.title.set_text(title)


def get_stat_and_loss_aug(model,
                          x,
                          y,
                          sample_weight: Optional[np.ndarray] = None,
                          batch_size=4096):
  """A helper function to get the statistics and losses.

  Here we get the statistics and losses for the original and
  horizontally flipped image, as we are going to train the model with
  random horizontal flip.

  Args:
    model: model to make prediction
    x: samples
    y: true labels of samples (integer valued)
    sample_weight: a vector of weights of shape (n_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.
    batch_size: the batch size for model.predict

  Returns:
    the statistics and cross-entropy losses
  """
  losses, stat = [], []
  for data in [x, x[:, :, ::-1, :]]:
    prob = amia.convert_logit_to_prob(
        model.predict(data, batch_size=batch_size))
    losses.append(utils.log_loss(y, prob, sample_weight=sample_weight))
    stat.append(
        amia.calculate_statistic(
            prob, y, sample_weight=sample_weight, convert_to_prob=False))
  return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)


def main(unused_argv):
  del unused_argv  # unused argument
  seed = 123
  np.random.seed(seed)
  if _MODEL_DIR.value and not os.path.exists(_MODEL_DIR.value):
    os.mkdir(_MODEL_DIR.value)

  # Load data.
  x, y = load_cifar10()
  # Sample weights are set to `None` by default, but can be changed here.
  sample_weight = None
  n = x.shape[0]

  # Train the target and shadow models. We will use one of the model in `models`
  # as target and the rest as shadow.
  # Here we use the same architecture and optimizer. In practice, they might
  # differ between the target and shadow models.
  in_indices = []  # a list of in-training indices for all models
  stat = []  # a list of statistics for all models
  losses = []  # a list of losses for all models
  for i in range(_NUM_SHADOWS.value + 1):
    if _MODEL_DIR.value:
      model_path = os.path.join(
          _MODEL_DIR.value,
          f'model{i}_lr{_LR.value}_b{_BATCH.value}_e{_EPOCHS.value}_sd{seed}.h5'
      )

    # Generate a binary array indicating which example to include for training
    in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))

    model = small_cnn()
    if _MODEL_DIR.value and os.path.exists(model_path):  # Load if exists
      model(x[:1])  # use this to make the `load_weights` work
      model.load_weights(model_path)
      print(f'Loaded model #{i} with {in_indices[-1].sum()} examples.')
    else:  # Otherwise, train the model
      model.compile(
          optimizer=tf.keras.optimizers.SGD(_LR.value, momentum=0.9),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])
      model.fit(
          x[in_indices[-1]],
          y[in_indices[-1]],
          validation_data=(x[~in_indices[-1]], y[~in_indices[-1]]),
          epochs=_EPOCHS.value,
          batch_size=_BATCH.value,
          verbose=2)
      if _MODEL_DIR.value:
        model.save_weights(model_path)
      print(f'Trained model #{i} with {in_indices[-1].sum()} examples.')

    # Get the statistics of the current model.
    s, l = get_stat_and_loss_aug(model, x, y, sample_weight)
    stat.append(s)
    losses.append(l)

    # Avoid OOM
    tf.keras.backend.clear_session()
    gc.collect()

  # Now we do MIA for each model
  for idx in range(_NUM_SHADOWS.value + 1):
    print(f'Target model is #{idx}')
    stat_target = stat[idx]  # statistics of target model, shape (n, k)
    in_indices_target = in_indices[idx]  # ground-truth membership, shape (n,)

    # `stat_shadow` contains statistics of the shadow models, with shape
    # (num_shadows, n, k). `in_indices_shadow` contains membership of the shadow
    # models, with shape (num_shadows, n). We will use them to get a list
    # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
    # (m, k) array, for m being the number of shadow models trained with
    # (resp. without) the j-th example, and k being the number of augmentations
    # (2 in our case).
    stat_shadow = np.array(stat[:idx] + stat[idx + 1:])
    in_indices_shadow = np.array(in_indices[:idx] + in_indices[idx + 1:])
    stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(n)]
    stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(n)]

    # Compute the scores and use them for MIA
    scores = amia.compute_score_lira(
        stat_target, stat_in, stat_out, fix_variance=True)
    attack_input = AttackInputData(
        loss_train=scores[in_indices_target],
        loss_test=scores[~in_indices_target],
        sample_weight_train=sample_weight,
        sample_weight_test=sample_weight)
    result_lira = mia.run_attacks(attack_input).single_attack_results[0]
    print('Advanced MIA attack with Gaussian:',
          f'auc = {result_lira.get_auc():.4f}',
          f'adv = {result_lira.get_attacker_advantage():.4f}')

    # We also try using `compute_score_offset` to compute the score. We take
    # the negative of the score, because higher statistics corresponds to higher
    # probability for in-training, which is the opposite of loss.
    scores = -amia.compute_score_offset(stat_target, stat_in, stat_out)
    attack_input = AttackInputData(
        loss_train=scores[in_indices_target],
        loss_test=scores[~in_indices_target],
        sample_weight_train=sample_weight,
        sample_weight_test=sample_weight)
    result_offset = mia.run_attacks(attack_input).single_attack_results[0]
    print('Advanced MIA attack with offset:',
          f'auc = {result_offset.get_auc():.4f}',
          f'adv = {result_offset.get_attacker_advantage():.4f}')

    # Compare with the baseline MIA using the loss of the target model
    loss_target = losses[idx][:, 0]
    attack_input = AttackInputData(
        loss_train=loss_target[in_indices_target],
        loss_test=loss_target[~in_indices_target],
        sample_weight_train=sample_weight,
        sample_weight_test=sample_weight)
    result_baseline = mia.run_attacks(attack_input).single_attack_results[0]
    print('Baseline MIA attack:', f'auc = {result_baseline.get_auc():.4f}',
          f'adv = {result_baseline.get_attacker_advantage():.4f}')

  # Plot and save the AUC curves for the three methods.
  _, ax = plt.subplots(1, 1, figsize=(5, 5))
  for res, title in zip([result_baseline, result_lira, result_offset],
                        ['baseline', 'LiRA', 'offset']):
    label = f'{title} auc={res.get_auc():.4f}'
    mia_plotting.plot_roc_curve(
        res.roc_curve,
        functools.partial(plot_curve_with_area, ax=ax, label=label))
  plt.legend()
  plt.savefig('advanced_mia_demo.png')


if __name__ == '__main__':
  app.run(main)

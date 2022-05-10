# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import os
import numpy as np
import matplotlib.pyplot as plt
import functools

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# from mi_lira_2021
from plot import sweep, load_data, generate_ours, generate_global


def do_plot_all(fn, keep, scores, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using one model as test model and the rest to train,
    with a full leave-one-out cross-validation.
    """

    all_predictions = []
    all_answers = []
    for i in range(0, len(keep)):
      mask = np.zeros(len(keep), dtype=bool)
      mask[i:i+1] = True
      prediction, answers = fn(keep[~mask],
                               scores[~mask],
                               keep[mask],
                               scores[mask])
      all_predictions.extend(prediction)
      all_answers.extend(answers)

    fpr, tpr, auc, acc = sweep_fn(np.array(all_predictions),
                                  np.array(all_answers, dtype=bool))

    low = tpr[np.where(fpr < .001)[0][-1]]
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%(legend, auc, acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f' % auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f' % acc

    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return acc, auc


def fig_fpr_tpr(poison_mask, scores, keep):

    plt.figure(figsize=(4, 3))

    # evaluate LiRA on the points that were not targeted by poisoning
    do_plot_all(functools.partial(generate_ours, fix_variance=True),
                keep[:, ~poison_mask], scores[:, ~poison_mask],
                "No poison (LiRA)\n",
                metric='auc',
    )

    # evaluate the global-threshold attack on the points that were not targeted by poisoning
    do_plot_all(generate_global,
                keep[:, ~poison_mask], scores[:, ~poison_mask],
                "No poison (Global threshold)\n",
                metric='auc', ls="--", c=plt.gca().lines[-1].get_color()
                )

    # evaluate LiRA on the points that were targeted by poisoning
    do_plot_all(functools.partial(generate_ours, fix_variance=True),
                keep[:, poison_mask], scores[:, poison_mask],
                "With poison (LiRA)\n",
                metric='auc',
                )

    # evaluate the global-threshold attack on the points that were targeted by poisoning
    do_plot_all(generate_global,
                keep[:, poison_mask], scores[:, poison_mask],
                "With poison (Global threshold)\n",
                metric='auc', ls="--", c=plt.gca().lines[-1].get_color()
    )

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-3, 1)
    plt.ylim(1e-3, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig("/tmp/fprtpr.png")
    plt.show()


if __name__ == '__main__':
  logdir = "exp/cifar10/"
  scores, keep = load_data(logdir)
  poison_pos = np.load(os.path.join(logdir, "poison_pos.npy"))
  poison_mask = np.zeros(scores.shape[1], dtype=bool)
  poison_mask[poison_pos] = True
  fig_fpr_tpr(poison_mask, scores, keep)

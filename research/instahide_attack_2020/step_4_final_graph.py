# Copyright 2020 Google LLC
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

import multiprocessing as mp
import pickle
import random
import numpy as np


labels = np.load("data/label.npy")
nextgraph = np.load("data/nextgraph.npy")

assigned = [[] for _ in range(5000)]
lambdas = [[] for _ in range(5000)]
for i in range(100):
    order = (np.argsort(nextgraph[:,i]))
    correct = (labels[order[:20]]>0).sum(axis=0).argmax()

    # Let's create the final graph
    # Instead of doing a full bipartite matching, let's just greedily
    # choose the closest 80 candidates for each encoded image to pair
    # together can call it a day.
    # This is within a percent or two of doing that, and much easier.

    # Also record the lambdas based on which image it coresponds to,
    # but if they share a label then just guess it's an even 50/50 split.

    
    for x in order[:80]:
        if labels[x][correct] > 0 and len(assigned[x]) < 2:
            assigned[x].append(i)
            if np.sum(labels[x]>0) == 1:
                # the same label was mixed in twice. punt.
                lambdas[x].append(labels[x][correct]/2)
            else:
                lambdas[x].append(labels[x][correct])

np.save("data/predicted_pairings_80.npy", assigned)
np.save("data/predicted_lambdas_80.npy", lambdas)

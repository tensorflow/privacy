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
import collections
import numpy as np

def score(subset):
    sub = graph[subset]
    sub = sub[:,subset]
    return np.sum(sub)

def run(v, return_scores=False):
    if isinstance(v, int):
        v = [v]
    scores = []
    for _ in range(100):
        keep = graph[v,:]
        next_value = np.sum(keep,axis=0)
        to_add = next_value.argsort()
        to_add = [x for x in to_add if x not in v]
        if _ < 1:
            v.append(to_add[random.randint(0,10)])
        else:
            v.append(to_add[0])
        if return_scores:
            scores.append(score(v)/len(keep))
    if return_scores:
        return v, scores
    else:
        return v

def make_many_clusters():
    # Compute clusters of 100 examples that probably correspond to some original image
    p = mp.Pool(mp.cpu_count())
    s = p.map(run, range(2000))
    return s


def downselect_clusters(s):
    # Right now we have a lot of clusters, but they probably overlap. Let's remove that.
    # We want to find disjoint clusters, so we'll greedily add them until we have
    # 100 distjoint clusters.
    
    ss = [set(x) for x in s]
    
    keep = []
    keep_set = []
    for iteration in range(2):
        for this_set in s:
            # MAGIC NUMBERS...!
            # We want clusters of size 50 because it works
            # Except on iteration 2 where we'll settle for 25 if we haven't
            # found clusters with 50 neighbors that work.
            cur = set(this_set[:50 - 25*iteration])
            intersections = np.array([len(cur & x) for x in ss])
            good = np.sum(intersections==50)>2
            # Good means that this cluster isn't a fluke and some other cluster
            # is like this one.
            if good or iteration == 1:
                print("N")
                # And also make sure we haven't found this cluster (or one like it).
                already_found = np.array([len(cur & x) for x in keep_set])
                if np.all(already_found<len(cur)/2):
                    print("And is new")
                    keep.append(this_set)
                    keep_set.append(set(this_set))
            if len(keep) == 100:
                break
        print("Found", len(keep))
        if len(keep) == 100:
            break

    # Keep should now have 100 items.
    # If it doesn't go and change the 2000 in make_many_clusters to a bigger number.
    return keep

if __name__ == "__main__":
    graph = np.load("data/graph.npy")
    np.save("data/many_clusters",make_many_clusters())
    np.save("data/100_clusters", downselect_clusters(np.load("data/many_clusters.npy")))

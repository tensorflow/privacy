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

"""
Create the improved graph mapping each encoded image to an original image.
"""

import objax
import numpy as np
import jax.numpy as jn
import functools
import os
import random

from objax.zoo import wide_resnet


def setup():
    global model
    class DoesUseSame(objax.Module):
        def __init__(self):
            fn = functools.partial(wide_resnet.WideResNet, depth=28, width=6)
            self.model = fn(3*4,2)
            
            model_vars = self.model.vars()
            self.ema = objax.optimizer.ExponentialMovingAverage(model_vars, momentum=0.999, debias=True)
    
            
            def predict_op(x,y):
                # The model takes SEVERAL images and checks if they all correspond
                # to the same original image.
                # Guaranteed that the first N-1 all do, the test is if the last does.
                xx = jn.concatenate([jn.abs(x),
                                     jn.abs(y)],
                                    axis=1)
                return self.model(xx, training=False)
            
            self.predict = objax.Jit(self.ema.replace_vars(predict_op), model_vars + self.ema.vars())
    
    model = DoesUseSame()
    checkpoint = objax.io.Checkpoint("models/step2/", keep_ckpts=5, makedir=True)
    start_epoch, last_ckpt = checkpoint.restore(model.vars())

def step2():
    global v, n, u, nextgraph

    # Start out by loading the encoded images
    n = np.load("data/encryption.npy")
    n = np.transpose(n, (0,3,1,2))

    # Then load the graph with 100 cluster-centers.
    keep = np.array(np.load("data/100_clusters.npy", allow_pickle=True))
    graph = np.load("data/graph.npy")
    

    # Now we're going to record the distance to each of the cluster centers
    # from every encoded image, so that we can do the matching.

    # To do that, though, first we need to choose the cluster centers.
    # Start out by choosing the best cluster centers.

    distances = []
    
    for x in keep:
        this_set = x[:50]
        use_elts = graph[this_set]
        distances.append(np.sum(use_elts,axis=0))
    distances = np.array(distances)
    
    ds = np.argsort(distances, axis=1)

    # Now we record the "prototypes" of each cluster center.
    # We just need three, more might help a little bit but not much.
    # (And then do that ten times, so we can average out noise
    # with respect to which cluster centers we picked.)
    
    prototypes = []
    for _ in range(10):
        ps = []
        # choose 3 random samples from each set
        for i in range(3):
            ps.append(n[ds[:,random.randint(0,20)]])
        prototypes.append(np.concatenate(ps,1))
    prototypes = np.concatenate(prototypes,0)

    # Finally compute the distances from each node to each cluster center.
    nextgraph = []
    for i in range(5000):
        out = model.predict(prototypes, np.tile(n[i:i+1], (1000,1,1,1)))
        out = out.reshape((10, 100, 2))
        
        v = np.sum(out,axis=0)
        v = v[:,0] - v[:,1]
        v = np.array(v)
        nextgraph.append(v)

    np.save("data/nextgraph.npy", nextgraph)

    
if __name__ == "__main__":
    setup()
    step2()

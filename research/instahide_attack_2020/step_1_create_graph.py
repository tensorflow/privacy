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
Create the similarity graph given the encoded images by running the similarity
neural network over all pairs of images.
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
            self.model = fn(6,2)
            
            model_vars = self.model.vars()
            self.ema = objax.optimizer.ExponentialMovingAverage(model_vars, momentum=0.999, debias=True)
    

            def predict_op(x,y):
                # The model takes the two images and checks if they correspond
                # to the same original image.
                xx = jn.concatenate([jn.abs(x),
                                     jn.abs(y)],
                                    axis=1)
                return self.model(xx, training=False)
            
            self.predict = objax.Jit(self.ema.replace_vars(predict_op), model_vars + self.ema.vars())
            self.predict_fast = objax.Parallel(self.ema.replace_vars(predict_op), model_vars + self.ema.vars())
    
    model = DoesUseSame()
    checkpoint = objax.io.Checkpoint("models/step1/", keep_ckpts=5, makedir=True)
    start_epoch, last_ckpt = checkpoint.restore(model.vars())


def doall():
    global graph
    n = np.load("data/encryption.npy")
    n = np.transpose(n, (0,3,1,2))

    # Compute the similarity between each encoded image and all others
    # This is n^2 work but should run fairly quickly, especially given
    # more than one GPU. Otherwise about an hour or so.
    graph = []
    with model.vars().replicate():
        for i in range(5000):
            print(i)
            v = model.predict_fast(np.tile(n[i:i+1], (5000,1,1,1)), n)
            graph.append(np.array(v[:,0]-v[:,1]))
    graph = np.array(graph)
    np.save("data/graph.npy", graph)

    
if __name__ == "__main__":
    setup()
    doall()

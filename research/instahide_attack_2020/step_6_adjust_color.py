# Copyright 2020 Google LLC
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

"""
Fix the color curves. Use a pre-trained "neural network" with <100 weights.
Visually this helps a lot, even if it's not doing much of anything in pactice.
"""

import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import jax.numpy as jn

import objax

# Our extremely complicated neural network to re-color the images.
# Takes one pixel at a time and fixes the color of that pixel.
model = objax.nn.Sequential([objax.nn.Linear(3, 10),
                             objax.functional.relu,
                             objax.nn.Linear(10, 3)
                             ])

# These are the weights.
weights = [[-0.09795442, -0.26434848, -0.24964345, -0.11450608, 0.6797288, -0.48435465,
            0.45307165, -0.31196147, -0.33266315, 0.20486055],
           [[-0.9056427, 0.02872663, -1.5114126, -0.41024876, -0.98195165, 0.1143966,
             0.6763464, -0.58654785, -1.797063, -0.2176538, ],
            [ 1.1941166, 0.15515928, 1.1691351, -0.7256186, 0.8046044, 1.3127686,
              -0.77297133, -1.1761239, 0.85841715, 0.95545965],
            [ 0.20092924, 0.57503146, 0.22809981, 1.5288007, -0.94781816, -0.68305916,
              -0.5245211, 1.4042739, -0.00527458, -1.1462274, ]],
           [0.15683544, 0.22086962, 0.33100453],
           [[ 7.7239674e-01, 4.0261227e-01, -9.6466336e-03],
            [-2.2159107e-01, 1.5123411e-01, 3.4485441e-01],
            [-1.7618114e+00, -7.1886492e-01, -4.6467595e-02],
            [ 6.9419539e-01, 6.2531930e-01, 7.2271496e-01],
            [-1.1913675e+00, -6.7755884e-01, -3.5114303e-01],
            [ 4.8022485e-01, 1.7145030e-01, 7.4849324e-04],
            [ 3.8332436e-02, -7.0614147e-01, -5.5127507e-01],
            [-1.0929481e+00, -1.0268525e+00, -7.0265180e-01],
            [ 1.4880739e+00, 7.1450096e-01, 2.9102692e-01],
            [ 7.2846663e-01,  7.1322352e-01, -1.7453632e-01]]]
           
for i,(k,v) in enumerate(model.vars().items()):
    v.assign(jn.array(weights[i]))

# Do all of the re-coloring
predict = objax.Jit(lambda x: model(x, training=False),
                    model.vars())

out = model(np.load("data/private_raw.npy"))
np.save("data/private.npy", out)

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
The final recovery happens here. Given the graph, reconstruct images.
"""


import json
import numpy as np
import jax.numpy as jn
import jax
import collections
from PIL import Image

import jax.example_libraries.optimizers

import matplotlib.pyplot as plt

def toimg(x):
    #x = np.transpose(x,(1,2,0))
    print(x.shape)
    img = (x+1)*127.5
    return Image.fromarray(np.array(img,dtype=np.uint8))



def explained_variance(I, private_images, lambdas, encoded_images, public_to_private, return_mat=False):
    # private images: 100x32x32x3
    # encoded images: 5000x32x32x3

    public_to_private = jax.nn.softmax(public_to_private,axis=-1)

    # Compute the components from each of the images we know should map onto the same original image.
    component_1 = jn.dot(public_to_private[0], private_images.reshape((100,-1))).reshape((5000,32,32,3))
    component_2 = jn.dot(public_to_private[1], private_images.reshape((100,-1))).reshape((5000,32,32,3))

    # Now combine them together to get the variance we can explain
    merged = component_1 * lambdas[:,0][:,None,None,None] + component_2 * lambdas[:,1][:,None,None,None]

    # And now get the variance we can't explain.
    # This is the contribution of the public images.
    # We want this value to be small.
    
    def keep_smallest_abs(xx1, xx2):
        t = 0
        which = (jn.abs(xx1+t) < jn.abs(xx2+t)) + 0.0
        return xx1 * which + xx2 * (1-which)
        
    xx1 = jn.abs(encoded) - merged
    xx2 = -(jn.abs(encoded) + merged)

    xx = keep_smallest_abs(xx1, xx2)
    unexplained_variance = xx
    

    if return_mat:
        return unexplained_variance, xx1, xx2
    
    extra = (1-jn.abs(private_images)).mean()*.05
    
    return extra + (unexplained_variance**2).mean()

def setup():
    global private, imagenet40, encoded, lambdas, using, real_using, pub_using

    # Load all the things we've made.
    encoded = np.load("data/encryption.npy")
    labels = np.load("data/label.npy")
    using = np.load("data/predicted_pairings_80.npy", allow_pickle=True)
    lambdas = list(np.load("data/predicted_lambdas_80.npy", allow_pickle=True))
    for x in lambdas:
        while len(x) < 2:
            x.append(0)
    lambdas = np.array(lambdas)

    # Construct the mapping
    public_to_private_new = np.zeros((2, 5000, 100))
    
    cs = [0]*100
    for i,row in enumerate(using):
        for j,b in enumerate(row[:2]):
            public_to_private_new[j, i, b] = 1e9
            cs[b] += 1
    using = public_to_private_new

def loss(private, lams, I):
    return explained_variance(I, private, lams, jn.array(encoded), jn.array(using))

def make_loss():
    global vg
    vg = jax.jit(jax.value_and_grad(loss, argnums=(0,1)))

def run():
    priv = np.zeros((100,32,32,3))
    uusing = np.array(using)
    lams = np.array(lambdas)

    # Use Adam, because thinking hard is overrated we have magic pixie dust.
    init_1, opt_update_1, get_params_1 = \
        jax.example_libraries.optimizers.adam(.01)
    @jax.jit
    def update_1(i, opt_state, gs):
        return opt_update_1(i, gs, opt_state)
    opt_state_1 = init_1(priv)

    # 1000 iterations of gradient descent is probably enough
    for i in range(1000):
        value, grad = vg(priv, lams, i)

        if i%100 == 0:
            print(value)

            var,_,_ = explained_variance(0, priv, jn.array(lambdas), jn.array(encoded), jn.array(using),
                                     return_mat=True)
            print('unexplained min/max', var.min(), var.max())
        opt_state_1 = update_1(i, opt_state_1, grad[0])
        priv = opt_state_1.packed_state[0][0]

    priv -= np.min(priv, axis=(1,2,3), keepdims=True)
    priv /= np.max(priv, axis=(1,2,3), keepdims=True)
    priv *= 2
    priv -= 1

    # Finally save the stored values
    np.save("data/private_raw.npy", priv)

    
if __name__ == "__main__":
    setup()
    make_loss()
    run()

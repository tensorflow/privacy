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

import sys
import numpy as np
import os
import multiprocessing as mp


def load_one(base):
    """
    This loads a  logits and converts it to a scored prediction.
    """
    root = os.path.join(logdir,base,'logits')
    if not os.path.exists(root): return None

    if not os.path.exists(os.path.join(logdir,base,'scores')):
        os.mkdir(os.path.join(logdir,base,'scores'))
    
    for f in os.listdir(root):
        try:
            opredictions = np.load(os.path.join(root,f))
        except:
            print("Fail")
            continue

        ## Be exceptionally careful.
        ## Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions/np.sum(predictions,axis=3,keepdims=True)

        COUNT = predictions.shape[0]
        #  x num_examples x num_augmentations x logits
        y_true = predictions[np.arange(COUNT),:,:,labels[:COUNT]]
        print(y_true.shape)

        print('mean acc',np.mean(predictions[:,0,0,:].argmax(1)==labels[:COUNT]))
        
        predictions[np.arange(COUNT),:,:,labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=3)

        logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45))

        np.save(os.path.join(logdir, base, 'scores', f), logit)


def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [x for x in os.listdir(logdir) if 'exp' in x])


logdir = sys.argv[1]
labels = np.load(os.path.join(logdir,"y_train.npy"))
load_stats()

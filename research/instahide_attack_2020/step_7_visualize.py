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
Given the private images, draw them in a 100x100 grid for visualization.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

p = np.load("data/private.npy")

def toimg(x):
    print(x.shape)
    img = (x+1)*127.5
    img = np.clip(img, 0, 255)
    img = np.reshape(img, (10, 10, 32, 32, 3))
    img = np.concatenate(img, axis=2)
    img = np.concatenate(img, axis=0)
    img = Image.fromarray(np.array(img,dtype=np.uint8))
    return img

toimg(p).save("data/reconstructed.png")


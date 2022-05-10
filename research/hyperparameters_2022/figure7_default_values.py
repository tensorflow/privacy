# Copyright 2022, The TensorFlow Privacy Authors.
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
"""Default values for generating Figure 7."""

import json
import numpy as np

orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
          list(range(5, 64)) + [128, 256, 512])
rdp = [
    2.04459751e-01, 2.45818210e-01, 2.87335988e-01, 3.29014798e-01,
    3.70856385e-01, 4.12862542e-01, 4.97375951e-01, 5.82570265e-01,
    6.68461534e-01, 7.55066706e-01, 8.42403732e-01, 1.01935100e+00,
    1.19947313e+00, 1.38297035e+00, 1.57009549e+00, 1.76124790e+00,
    1.95794503e+00, 2.19017390e+00, 4.48407479e+00, 3.08305394e+02,
    4.98610133e+03, 1.11363692e+04, 1.72590079e+04, 2.33487231e+04,
    2.94091123e+04, 3.54439803e+04, 4.14567914e+04, 4.74505356e+04,
    5.34277419e+04, 5.93905358e+04, 6.53407051e+04, 7.12797586e+04,
    7.72089762e+04, 8.31294496e+04, 8.90421151e+04, 9.49477802e+04,
    1.00847145e+05, 1.06740819e+05, 1.12629335e+05, 1.18513163e+05,
    1.24392717e+05, 1.30268362e+05, 1.36140424e+05, 1.42009194e+05,
    1.47874932e+05, 1.53737871e+05, 1.59598221e+05, 1.65456171e+05,
    1.71311893e+05, 1.77165542e+05, 1.83017260e+05, 1.88867175e+05,
    1.94715404e+05, 2.00562057e+05, 2.06407230e+05, 2.12251015e+05,
    2.18093495e+05, 2.23934746e+05, 2.29774840e+05, 2.35613842e+05,
    2.41451813e+05, 2.47288808e+05, 2.53124881e+05, 2.58960080e+05,
    2.64794449e+05, 2.70628032e+05, 2.76460867e+05, 2.82292992e+05,
    2.88124440e+05, 6.66483142e+05, 1.41061455e+06, 2.89842152e+06
]
with open("lr_acc.json", "r") as dict_f:
  lr_acc = json.load(dict_f)
num_trials = 1000
lr_rates = np.logspace(np.log10(1e-4), np.log10(1.), num=1000)[-400:]
gammas = np.asarray(
    [1e-07, 8e-06, 1e-04, 0.00024, 0.0015, 0.0035, 0.025, 0.05, 0.1, 0.2, 0.5])
non_private_acc = 0.9594

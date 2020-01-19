# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License,  Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r"""This code applies the Dual and Central Limit
Theorem (CLT) to estimate privacy budget of an iterated subsampled
Gaussian Mechanism (by either uniform or Poisson subsampling).
"""


import numpy as np
from scipy.stats import norm
from scipy import optimize

# Total number of examples:N
# batch size:batch_size
# Noise multiplier for DP-SGD/DP-Adam:noise_multiplier
# current epoch:epoch
# Target delta:delta

def compute_mu_uniform(epoch, noise_multi, N, batch_size):
    '''Compute mu from uniform subsampling'''
    T = epoch*N/batch_size
    c = batch_size*np.sqrt(T)/N
    return np.sqrt(2)*c*np.sqrt(np.exp(noise_multi**(-2))*\
                   norm.cdf(1.5/noise_multi)+3*norm.cdf(-0.5/noise_multi)-2)

def compute_mu_Poisson(epoch, noise_multi, N, batch_size):
    '''Compute mu from Poisson subsampling'''
    T = epoch*N/batch_size
    return np.sqrt(np.exp(noise_multi**(-2))-1)*np.sqrt(T)*batch_size/N

def delta_eps_mu(eps, mu):
    '''Dual between mu-GDP and (epsilon, delta)-DP'''
    return norm.cdf(-eps/mu+mu/2)-np.exp(eps)*norm.cdf(-eps/mu-mu/2)

def eps_from_mu(mu, delta):
    '''inverse Dual'''
    def f(x):
        '''reversely solving dual'''
        return delta_eps_mu(x, mu) - delta
    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

def compute_eps_uniform(epoch, noise_multi, N, batch_size, delta):
    '''inverse Dual of uniform subsampling'''
    return eps_from_mu(compute_mu_uniform(epoch, noise_multi, N, batch_size), delta)

def compute_eps_Poisson(epoch, noise_multi, N, batch_size, delta):
    '''inverse Dual of Poisson subsampling'''
    return eps_from_mu(compute_mu_Poisson(epoch, noise_multi, N, batch_size), delta)

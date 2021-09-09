from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from absl import app
from absl import flags

import numpy as np

import tensorflow.compat.v1 as tf

def cactus_sample(stddev=0.25,n=200,r=0.9, size=[2,4]):

  filename=('f_cactus_cdf_v%.2f.csv' %stddev)
  with open(filename, newline='') as csvfile:
      cdf = np.array(list(csv.reader(csvfile)), dtype=np.float32).reshape(-1)

  filename=('f_cactus_x_v%.2f.csv' %stddev)
  with open(filename, newline='') as csvfile:
      x = np.array(list(csv.reader(csvfile)), dtype=np.float32).reshape(-1)

  l=len(x)

  result=np.zeros(size)
  for a in range(len(result)):
    for b in range(len(result[0])):
      rand=np.random.uniform(0,1,1)
      i = np.argwhere(rand<cdf)[0]
      s = x[i]
      if i==0 or i==l-1:
        j=np.floor(np.log(rand)/log(r))
        if i==0:
          s = s-j/n
        else:
          s = s+j/n
      result[a][b]=s+(rand-0.5)/n
  return result

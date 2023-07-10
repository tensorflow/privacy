"""print the achievable error of different algorithms"""
import json
import os
import numpy as np

RESULTS_PATH = './src/results/'
excess_loss = {}
opt_algs = ["DPGD", "DN-Hess-add", "DN-UB-add", "DN-Hess-clip",
             "DN-UB-clip", "private-newton"]
for filename in os.listdir(RESULTS_PATH):
    f = os.path.join(RESULTS_PATH, filename)
    with open(f, encoding='utf-8') as json_file:
        data = json.load(json_file)
        for alg in data.keys():
            if alg in opt_algs:
                loss_avg = np.array(data[alg]['loss_avg'])
                loss_std = np.array(data[alg]['loss_std'])
                clock_time = np.array(data[alg]['clock_time_avg'])
                print('optimization algorithm: ',alg)
                print('excess loss: ' + str(loss_avg[-1]))
                print('run time: ' + str(clock_time[-1]) + '(sec)')
                print('-----')

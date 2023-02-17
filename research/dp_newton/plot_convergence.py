import json
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

alg_keys = {
    "DPGD": 'DP-GD',
    "DN-Hess-add": 'HESS-ADD',
    "DN-UB-add": 'QU-ADD',
    "DN-Hess-clip": 'Hess-Clip',
    "DN-UB-clip": 'QU-Clip',
    "DPGD-Oracle": 'DPGD-Oracle',
    "private-newton": '[ABL21]'
    }

linestyle = {
    "DPGD": 'm--o',
    "DN-Hess-add": 'k-',
    "DN-UB-add": 'b--',
    "DN-Hess-clip": 'r',
    "DN-UB-clip": 'g',
    "DPGD-Oracle": 'c-',
    "private-newton": 'y--'
    }

facecolor = {
    "DPGD": 'magenta',
    "DN-Hess-add": 'black',
    "DN-UB-add": 'blue',
    "DN-Hess-clip": 'red',
    "DN-UB-clip": 'green',
    "DPGD-Oracle": 'cyan',
    "private-newton": 'yellow'
    }

directory = 'results/'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    datasetname = filename.split('_')[1]
    privacy_budget = filename.split('_')[3]
    plt.figure(1)
    with open(f) as json_file:
        data = json.load(json_file)
        for alg in data.keys():
            if alg == 'num-samples' or alg == 'obj-perturb' or alg == 'acc-best':
                continue
            else:
                loss_avg, loss_std = np.array(data[alg]['loss_avg']), np.array(data[alg]['loss_std'])
                iters_idx = np.arange(len(loss_avg))
                ax = plt.gca()
                plt.plot(iters_idx, loss_avg, linestyle[alg], label=alg_keys[alg], linewidth=3)
                plt.fill_between(iters_idx, loss_avg - loss_std, loss_avg + loss_std, facecolor=facecolor[alg], alpha=0.2)
                plt.legend(fontsize=16)
                plt.grid(True, which="both")
                ax.grid(alpha=0.2)
                plt.yscale('log')
                plt.yticks(fontsize=18)
                plt.ylabel("Excess Loss", fontsize=18)
                plt.xlabel("Iteration", fontsize=18)
                plt.xticks(fontsize=16)
                plt.savefig(str(datasetname)+'-'+str(privacy_budget)+'-convergence.pdf', bbox_inches='tight')
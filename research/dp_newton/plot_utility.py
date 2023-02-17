import json
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


alg_keys = {
    "DPGD": 'DPGD',
    "DN-Hess-add": 'HESS-ADD',
    "DN-UB-add": 'QU-ADD',
    "DN-Hess-clip": 'HESS-CLIP',
    "DN-UB-clip": 'QU-CLIP',
    "obj-perturb": 'AOP',
    "private-newton": '[ABL21]'
    }
alg_indx = {
    "DPGD": 0,
    "DN-Hess-add": 1,
    "DN-UB-add": 2,
    "DN-Hess-clip": 3,
    "DN-UB-clip": 4,
    "obj-perturb": 5,
    "private-newton": 6
    }
linestyle = {
    "DPGD": 'm-',
    "DN-Hess-add": 'k-',
    "DN-UB-add": 'b--',
    "DN-Hess-clip": 'r-',
    "DN-UB-clip": 'g-.',
    "private-newton": 'c-',
    "obj-perturb": 'y'
    }
facecolor = {
    "DPGD": 'magenta',
    "DN-Hess-add": 'black',
    "DN-UB-add": 'blue',
    "DN-Hess-clip": 'red',
    "DN-UB-clip": 'green',
    "private-newton": 'cyan',
    "obj-perturb": 'yellow'
}

directory = 'results/'


privacy_params = []
for filename in os.listdir(directory):
    privacy_budget = float(filename.split('_')[4])
    privacy_params.append(privacy_budget)

privacy_params = np.sort(np.array(privacy_params))
loss_val_summ = np.zeros((len(alg_keys), len(privacy_params)))
loss_std_summ = np.zeros((len(alg_keys), len(privacy_params)))

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    datasetname = filename.split('_')[1]
    privacy_budget = float(filename.split('_')[4])
    plt.clf()
    with open(f) as json_file:
        data = json.load(json_file)
        for alg in data.keys():
            if alg == 'acc-best' or alg == 'DPGD-Oracle':
                continue
            elif alg == 'num-samples':
                num_samples = int(data[alg])
            elif alg == 'obj-perturb':
                loss_val_summ[alg_indx[alg]][np.where(privacy_params == privacy_budget)[0][0]] = data[alg]['loss_avg']
                loss_std_summ[alg_indx[alg]][np.where(privacy_params == privacy_budget)[0][0]] = data[alg]['loss_std']
            else:
                loss_avg, loss_std = np.array(data[alg]['loss_avg']), np.array(data[alg]['loss_std'])
                loss_val_summ[alg_indx[alg]][np.where(privacy_params == privacy_budget)[0][0]] = loss_avg[-1]
                loss_std_summ[alg_indx[alg]][np.where(privacy_params == privacy_budget)[0][0]] = loss_std[-1]


for i in range(len(loss_val_summ)):
    alg = get_keys_from_value(alg_indx, i)[0]
    ax = plt.gca()
    fig = plt.figure(1)
    plt.plot(privacy_params, loss_val_summ[i], linestyle[alg], label=alg_keys[alg], linewidth=3)
    plt.fill_between(privacy_params, loss_val_summ[i] - loss_std_summ[i], loss_val_summ[i] + loss_std_summ[i], facecolor=facecolor[alg], alpha=0.4)
    plt.yscale('log')
    plt.legend(fontsize=16)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.grid(True, which="both")
    ax.grid(alpha=0.2)
    plt.ylabel("Excess Loss", fontsize=18)
    plt.xlabel(r"Privacy Budget $(\epsilon,\delta=n^{-2})$-DP", fontsize=18)
    plt.savefig(str(datasetname) + '-loss-privacybudget.pdf', bbox_inches='tight')

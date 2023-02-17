import numpy as np
import json
import argparse
from dataset_loader import Mydatasets
from approx_op import ApproximateMinimaPerturbationLR
from my_logistic_regression import MyLogisticRegression
from opt_algs  import newton, gd_priv, gd_priv_optls, private_newton, DoubleNoiseMech, CompareAlgs
from myutils import zcdp_to_eps, eps_to_zcdp


def helper_fun(datasetname,pb,num_rep,Tuning,privacy_type='zcdp'):
    """ This function is a helper function for running different algorithms

    datasetname = name of the dataset
    pb = a dictionary with the parameters
    num_rep = number of times we repeat the optimization algorithm to report the average
    Tuning = True or False exhustive search for finding the best min eigenvalue
    """
    datasets = Mydatasets()
    X,y,w_opt = getattr(datasets,datasetname)()
    dataset = X,y
    priv_param = pb["total"]
    num_samples = len(y)
    delta = (1.0/num_samples)**2
    if privacy_type == 'dp':
        total = pb["total"]
        rho_eq = eps_to_zcdp(total,delta)
        pb["total"] = rho_eq
        print("privacy constraint is DP!"+' equaivalent rho: '+str(pb["total"]))
    else:
        print("privacy constraint is zCDP!"+' equaivalent rho: '+str(pb["total"]))
    lr = MyLogisticRegression(X,y,reg=1e-8)
    dnm_hess_add = DoubleNoiseMech(lr,type_reg='add',hyper_tuning=False,curvature_info='hessian').update_rule
    dnm_ub_add = DoubleNoiseMech(lr,type_reg='add',hyper_tuning=False,curvature_info='ub').update_rule
    dnm_hess_clip = DoubleNoiseMech(lr,type_reg='clip',hyper_tuning=False,curvature_info='hessian').update_rule
    dnm_ub_clip = DoubleNoiseMech(lr,type_reg='clip',hyper_tuning=False,curvature_info='ub').update_rule
    amp = ApproximateMinimaPerturbationLR().run_classification
    acc_objpert = []
    loss_objpert = []
    eps = zcdp_to_eps(pb["total"],delta)
    if Tuning:
        dnm_hess_clip_ht = DoubleNoiseMech(lr,type_reg='clip',hyper_tuning=True,curvature_info='hessian').update_rule
    c = CompareAlgs(lr,dataset,w_opt,iters=pb["num_iteration"],pb=pb)
    for rep in range(num_rep):
        print(str(rep+1)+" expriment out of "+ str(num_rep))
        c.add_algo(gd_priv,"DPGD")
        c.add_algo(dnm_hess_add,"DN-Hess-add")
        c.add_algo(dnm_ub_add,"DN-UB-add")
        c.add_algo(private_newton,"private-newton")
        c.add_algo(dnm_hess_clip,"DN-Hess-clip")
        c.add_algo(dnm_ub_clip,"DN-UB-clip")
        c.add_algo(gd_priv_optls,"DPGD-Oracle")
        theta_objpert, _ = amp(X, y, eps, delta, lambda_param=None, l2_constraint=None) 
        acc_objpert.append(lr.accuracy(theta_objpert))
        loss_objpert.append(lr.loss_wor(theta_objpert)-lr.loss_wor(w_opt))
        
        if Tuning:
            c.add_algo(dnm_hess_clip_ht,"DN-Hess-clip-T")
        losses_dict = c.loss_vals()
        gradnorm_dict = c.gradnorm_vals()
        accuracy_dict = c.accuracy_vals()
        if rep == 0:
            losses_total = losses_dict
            gradnorm_total = gradnorm_dict
            accuracy_total = accuracy_dict
        else:
            for names in losses_total.keys():
                losses_total[names].extend(losses_dict[names])
                gradnorm_total[names].extend(gradnorm_dict[names])
                accuracy_total[names].extend(accuracy_dict[names])

    result = {}
    accuracy_wopt = c.accuracy_np()
    result['num-samples'] = num_samples
    result['acc-best'] = accuracy_wopt.tolist()
    result['obj-perturb'] = {} 
    result['obj-perturb']["loss_avg"] = np.mean(loss_objpert).tolist()
    result['obj-perturb']["loss_std"] = (np.std(loss_objpert) / np.sqrt(num_rep)).tolist()
    result['obj-perturb']["acc_avg"] = np.mean(acc_objpert).tolist()
    result['obj-perturb']["acc_std"] = (np.std(acc_objpert) / np.sqrt(num_rep)).tolist()
    for alg in losses_total.keys():
        losses = np.array(losses_total[alg])
        gradnorm = np.array(gradnorm_total[alg])
        acc = np.array(accuracy_total[alg])
        result[alg] = {}
        result[alg]["loss_avg"] = np.mean(losses, axis=0).tolist()
        result[alg]["loss_std"] = (np.std(losses, axis=0) / np.sqrt(num_rep)).tolist()
        result[alg]["gradnorm_avg"] = np.mean(gradnorm, axis=0).tolist()
        result[alg]["gradnorm_std"] = (np.std(gradnorm, axis=0) / np.sqrt(num_rep)).tolist()
        result[alg]["acc_avg"] = np.mean(acc, axis=0).tolist()
        result[alg]["acc_std"] = (np.std(acc, axis=0) / np.sqrt(num_rep)).tolist()

    json.dump(result, open("results/new_result_"+datasetname+"_"+str(priv_param)+"_"+privacy_type+"_"+str(pb["num_iteration"])+"_"+str(Tuning)+".txt", 'w'))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetname")
    parser.add_argument("total")
    parser.add_argument("numiter")
    parser.add_argument("tuning")
    parser.add_argument("privacy_type")
    args = parser.parse_args()
    datasetname = args.datasetname
    privacy_type = args.privacy_type # it is either 'dp' or 'zcdp'
    total = float(args.total)
    num_iter = int(args.numiter) 
    tuning = int(args.tuning)  # it is either 0 or 1, Tuning-->1 (True) or Tuning-->0 (False)
    pb = {
      "total": total,  # Total privacy budget
      "grad_frac": 0.75,  # Fraction of privacy budget for gradient vs search direction
      "num_iteration": num_iter
    }
    num_rep = 30 # the number of repetitions for averaging over the randomness 
    print("the dataset is ", str(datasetname))
    print('total is '+str(total)+' num_iter '+str(num_iter)+' tuning '+str(bool(tuning)))
    helper_fun(datasetname,pb,num_rep=num_rep,Tuning=bool(tuning),privacy_type=privacy_type)


if __name__ == '__main__':
    main()

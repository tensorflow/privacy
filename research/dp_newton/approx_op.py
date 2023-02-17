import math
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import sklearn.datasets
import urllib.request
import tensorflow as tf
import json
from tensorflow import keras
from op_utils.common import Algorithm
from op_utils.op_logistic_regression import LogisticRegression_OP
import os
from constraints import constrain_l2_norm


# objective perturbation code from 'Differentially Private Convex Optimization Benchmark' repo



def amp_run_classification(x, y, loss_func, grad_func,
                           epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=None,
                           l2_constraint=None, eps_frac=0.9,
                           eps_out_frac=0.01,
                           gamma=None, L=1, gamma_mult=1):
    
    n = x.shape[0]
    m = x.shape[1]
    lmbda = pow(L, 2)
    r = 2  # for GLMs
    beta = pow(L, 2)  # from psgd
    USE_LOWMEM = False
    # initial model
    x0 = np.zeros(shape=x.shape[1])

    # hard-code the split for obj/out
    delta_out_frac = eps_out_frac

    # strategy for split within obj
    if eps_frac is None:
        # old strategy
        # best = 0.796 + 0.149*np.exp(-3.435*epsilon)

        # "Strategy #1"
        best = min(0.88671 + 0.0186607 / (epsilon ** 0.372906), .99)

        # "Strategy #2"
        # best = 0.909994+0.0769162*np.exp(-9.41309*epsilon)

        eps_frac = max(best, 1 - 1 / epsilon + 0.001)

    # split the budget 3 ways
    eps_out = epsilon * eps_out_frac
    eps_obj = epsilon - eps_out
    eps_p = eps_frac * eps_obj

    delta_out = delta_out_frac * delta
    delta_obj = delta - delta_out

    # set the lower bound on regularization
    big_lambda = r * beta / (eps_obj - eps_p)

    # set gamma
    if gamma is None:
        if USE_LOWMEM:
            gamma = 1.0 / n
        else:
            gamma = 1.0 / (n ** 2)

    # enforce the constraint on eps_p
    if (eps_obj - eps_p) >= 1:
        return x0, gamma

    effective_gamma = gamma * gamma_mult

    # set the sensitivity
    sensitivity_obj = 2 * L / n
    sensitivity_out = n * gamma / big_lambda

    # set the std dev of noise for obj part
    std_dev_obj = sensitivity_obj * (1 + np.sqrt(2 * np.log(1 / delta_obj))) / eps_p
    std_dev_out = sensitivity_out * (1 + np.sqrt(2 * np.log(1 / delta_out))) / eps_out

    # generate the noise for obj part
    np.random.seed(ord(os.urandom(1)))
    noise_obj = np.random.normal(scale=std_dev_obj, size=x.shape[1])

    # generate the noise for out part
    noise_out = np.random.normal(scale=std_dev_out, size=x.shape[1])

    if l2_constraint is None:
        x0 = np.zeros(shape=x.shape[1])
    else:
        x0 = (np.random.rand(x.shape[1]) - .5) * 2 * l2_constraint

    def private_loss(theta, x, y):
        raw_loss = loss_func(theta, x, y)
        result = (raw_loss + ((big_lambda / (2 * n)) *
                              (np.linalg.norm(theta, ord=2) ** 2)) + \
                  (noise_obj.T @ theta)) * gamma_mult
        return result

    def private_gradient(theta, x, y, use_gamma_mult=True):
        raw_gradient = grad_func(theta, x, y)
        result = raw_gradient + ((big_lambda / n) * theta) + noise_obj
        if use_gamma_mult:
            result *= gamma_mult
        return result

    if USE_LOWMEM:
        c = 200
        opts = {'gtol': effective_gamma / c}
        result = minimize(private_loss, x0, (x, y), method='L-BFGS-B',
                          jac=private_gradient, options=opts)
        theta = result.x
        grad = private_gradient(theta, x, y)
        norm = np.linalg.norm(grad, ord=2)

        if norm <= effective_gamma:
            theta_mid = result.x
            return theta_mid + noise_out, gamma
        else:
            if effective_gamma < 1e-04:
                gamma_mult *= 10
            else:
                gamma_mult = 1
                gamma *= 2
            return amp_run_classification(x, y, loss_func, grad_func, epsilon, delta, lambda_param,
                                          learning_rate=learning_rate, num_iters=None, l2_constraint=l2_constraint,
                                          eps_frac=eps_frac, gamma=gamma, L=L, gamma_mult=gamma_mult)
    else:
        def constrain_theta(theta):
            theta = constrain_l2_norm(theta, l2_constraint)

        if l2_constraint is not None:
            cb = constrain_theta
        else:
            cb = None

        opts = {'gtol': effective_gamma, 'norm': 2}
        
        result = minimize(private_loss, x0, (x, y), method='BFGS',
                          jac=private_gradient, options=opts, callback=cb)
        # result = minimize(private_loss, x0, (x, y), method='L-BFGS-B',
        #                   jac=private_gradient, options=opts, callback=cb) 
        # # Just for KDD-CUP dataset don't forget to uncomment it                 
        theta = result.x
        grad = private_gradient(theta, x, y)
        norm = np.linalg.norm(grad, ord=2)
        print(result.success)
        if not result.success:
            if effective_gamma < 1e-04:
                gamma_mult *= 10
            else:
                gamma_mult = 1
                gamma *= 2

            return amp_run_classification(x, y, loss_func, grad_func, epsilon, delta, lambda_param,
                                          learning_rate=learning_rate, num_iters=None, l2_constraint=l2_constraint,
                                          eps_frac=eps_frac, gamma=gamma, L=L, gamma_mult=gamma_mult)
        else:
            orig_gamma = 1 / (n ** 2)
            orig_grad = private_gradient(theta, x, y, use_gamma_mult=False)
            orig_norm = np.linalg.norm(orig_grad, ord=2)

            theta_mid = result.x
            return theta_mid + noise_out, gamma


class ApproximateMinimaPerturbationLR(Algorithm):
    def run_classification(self, x, y, epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=None,
                           l2_constraint=None, eps_frac=0.9,
                           eps_out_frac=0.01,
                           gamma=None, L=1):
        return amp_run_classification(x, y, LogisticRegression_OP.loss, LogisticRegression_OP.gradient,
                                      epsilon, delta, lambda_param,
                                      learning_rate=learning_rate, num_iters=num_iters,
                                      l2_constraint=l2_constraint, eps_frac=eps_frac,
                                      eps_out_frac=eps_out_frac,
                                      gamma=gamma, L=L)

    def name():
        return "Approximate minima perturbation with scipy minimize LR"


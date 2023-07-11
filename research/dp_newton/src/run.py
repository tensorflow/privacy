# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""collections of helper function to run and compare different algorithms"""

# pylint: skip-file
# pyformat: disable

import argparse
import json
from dataset_loader import Mydatasets
from my_logistic_regression import MyLogisticRegression
import numpy as np
from opt_algs import CompareAlgs, DoubleNoiseMech, gd_priv, private_newton
from scipy.optimize import fsolve


def zcdp_to_eps(rho, delta):
  """ "

  conversion of zcdp gurantee to (eps,delta)-DP using the formula in Lemma 3.6
  of [BS16]
  rho : zCDP
  delta: delta in DP

  return eps
  """
  return rho + np.sqrt(4 * rho * np.log(np.sqrt(np.pi * rho) / delta))


def eps_to_zcdp(eps, delta):
  """ "

  conversion of (eps,delta) gurantee to rho-zCDP
  eps : eps in DP
  delta: delta in DP

  return rho
  """

  def func_root(rho_zcdp):
    return zcdp_to_eps(rho_zcdp, delta) - eps

  root = fsolve(func_root, x0=0.001)[-1]
  return root


def helper_fun(datasetname, alg_type, params_exp):
  """helper function for running different algorithms

  args:
      datasetname = dataset
      alg_type = type of the optimization algorithm
      params_exp = hyperparameters
  """
  feature_vecs, labels, w_opt = getattr(Mydatasets(), datasetname)()
  privacy_dp = params_exp["total"]
  params_exp["total"] = eps_to_zcdp(privacy_dp, (1.0 / len(labels)) ** 2)
  log_reg = MyLogisticRegression(feature_vecs, labels)
  alg_dict, filename_params = prepare_alg_dict(
      alg_type, datasetname, privacy_dp, params_exp, log_reg
  )
  compare_algs = CompareAlgs(log_reg, w_opt, params_exp)
  result = RunReleaseStats(compare_algs, alg_dict).summarize_stats()
  result["num-samples"] = len(labels)
  with open(
      "src/results/" + filename_params, "w", encoding="utf8"
  ) as json_file:
    json.dump(result, json_file)


def prepare_alg_dict(alg_type, datasetname, privacy_dp, params_exp, log_reg):
  """prepare update rule for algorithms and filename"""
  alg_dict = None
  filename_params = None
  if alg_type == "double_noise":
    filename_params = (
        "so_"
        + datasetname
        + "_"
        + str(privacy_dp)
        + "_"
        + "DP"
        + "_"
        + str(params_exp["num_iteration"])
        + "_"
        + str(params_exp["grad_frac"])
        + "_"
        + str(params_exp["trace_frac"])
        + "_"
        + str(params_exp["trace_coeff"])
        + ".txt"
    )
    dnm_hess_add = DoubleNoiseMech(
        log_reg, type_reg="add", curvature_info="hessian"
    ).update_rule
    dnm_ub_add = DoubleNoiseMech(
        log_reg, type_reg="add", curvature_info="ub"
    ).update_rule
    dnm_hess_clip = DoubleNoiseMech(
        log_reg, type_reg="clip", curvature_info="hessian"
    ).update_rule
    dnm_ub_clip = DoubleNoiseMech(
        log_reg, type_reg="clip", curvature_info="ub"
    ).update_rule
    alg_dict = {
        "DN-Hess-add": dnm_hess_add,
        "DN-Hess-clip": dnm_hess_clip,
        "DN-UB-clip": dnm_ub_clip,
        "DN-UB-add": dnm_ub_add,
    }
  elif alg_type == "dp_gd":
    filename_params = (
        "gd_"
        + datasetname
        + "_"
        + str(privacy_dp)
        + "_"
        + "DP"
        + "_"
        + str(params_exp["num_iteration"])
        + ".txt"
    )
    alg_dict = {"DPGD": gd_priv}
  elif alg_type == "damped_newton":
    filename_params = (
        "newton_"
        + datasetname
        + "_"
        + str(privacy_dp)
        + "_"
        + "DP"
        + "_"
        + str(params_exp["num_iteration"])
        + ".txt"
    )
    alg_dict = {"private-newton": private_newton}
  return alg_dict, filename_params


class RunReleaseStats:
  """Helpfer function to run different algorithms and store the results"""

  def __init__(self, compare_algs, algs_dict, num_rep=10):
    self.compare_algs = compare_algs
    self.algs_dict = algs_dict
    self.num_rep = num_rep
    self.losses = 0
    self.gradnorm = 0
    self.accuracy = 0
    self.wall_clock = 0

  def run_algs(self):
    """method to run different algorithms and store different stats"""
    for rep in range(self.num_rep):
      for alg_name, alg_update_rule in self.algs_dict.items():
        self.compare_algs.add_algo(alg_update_rule, alg_name)
      losses_dict = self.compare_algs.loss_vals()
      gradnorm_dict = self.compare_algs.gradnorm_vals()
      accuracy_dict = self.compare_algs.accuracy_vals()
      wall_clock_dict = self.compare_algs.wall_clock_alg()
      if rep == 0:
        self.losses = losses_dict
        self.gradnorm = gradnorm_dict
        self.accuracy = accuracy_dict
        self.wall_clock = wall_clock_dict
      else:
        for alg in self.losses:
          self.losses[alg].extend(losses_dict[alg])
          self.gradnorm[alg].extend(gradnorm_dict[alg])
          self.accuracy[alg].extend(accuracy_dict[alg])
          self.wall_clock[alg].extend(wall_clock_dict[alg])

  def summarize_stats(self):
    """method to summarize the results"""
    self.run_algs()
    result = {}
    result["acc-best"] = self.compare_algs.accuracy_np().tolist()
    for alg in self.losses:
      result[alg] = {}
      loss_avg = np.mean(np.array(self.losses[alg]), axis=0)
      loss_std = np.std(np.array(self.losses[alg]), axis=0)
      result[alg]["loss_avg"] = (loss_avg).tolist()
      result[alg]["loss_std"] = (loss_std / np.sqrt(self.num_rep)).tolist()
      gradnorm_avg = np.mean(np.array(self.gradnorm[alg]), axis=0)
      gradnorm_std = np.std(np.array(self.gradnorm[alg]), axis=0)
      result[alg]["gradnorm_avg"] = (gradnorm_avg).tolist()
      result[alg]["gradnorm_std"] = (gradnorm_std).tolist()
      acc_avg = np.mean(np.array(self.accuracy[alg]), axis=0)
      acc_std = np.std(np.array(self.accuracy[alg]), axis=0)
      result[alg]["acc_avg"] = (acc_avg).tolist()
      result[alg]["acc_std"] = (acc_std / np.sqrt(self.num_rep)).tolist()
      clocktime_avg = np.mean(np.array(self.wall_clock[alg]), axis=0)
      clocktime_std = np.std(np.array(self.wall_clock[alg]), axis=0)
      result[alg]["clock_time_avg"] = (clocktime_avg).tolist()
      result[alg]["clock_time_std"] = (
          clocktime_std / np.sqrt(self.num_rep)
      ).tolist()

    return result


def main():
  """main function"""
  parser = argparse.ArgumentParser()
  parser.add_argument("--datasetname")
  parser.add_argument("--alg_type")
  parser.add_argument("--total")
  parser.add_argument("--numiter")
  # double noise and newton
  parser.add_argument("--grad_frac")
  parser.add_argument("--trace_frac")
  parser.add_argument("--trace_coeff")
  args = parser.parse_args()
  datasetname = args.datasetname
  alg_type = args.alg_type
  total = float(args.total)
  num_iter = int(args.numiter)
  if alg_type == "double_noise":
    grad_frac = float(args.grad_frac)
    trace_frac = float(args.trace_frac)
    trace_coeff = float(args.trace_coeff)
    hyper_parameters = {
        "total": total,
        "grad_frac": grad_frac,
        "trace_frac": trace_frac,
        "trace_coeff": trace_coeff,
        "num_iteration": num_iter,
    }
  elif alg_type == "dp_gd":
    hyper_parameters = {"total": total, "num_iteration": num_iter}
  elif alg_type == "damped_newton":
    grad_frac = float(args.grad_frac)
    hyper_parameters = {
        "total": total,
        "num_iteration": num_iter,
        "grad_frac": grad_frac,
    }
  else:
    raise ValueError("no such optmization algorithm exists")
  print(
      "optimization algorithm "
      + alg_type
      + ","
      + "dataset name: "
      + datasetname
  )
  helper_fun(datasetname, alg_type, hyper_parameters)


if __name__ == "__main__":
  main()

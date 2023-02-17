import numpy as np
from scipy.optimize import fsolve

def zcdp_to_eps(rho,delta):
  """"
    conversion of zcdp gurantee to (eps,delta)-DP
    rho : zCDP 
    delta: delta in DP

    return eps
  """
  return rho + np.sqrt(4 * rho * np.log(np.sqrt(np.pi * rho)/delta))


def eps_to_zcdp(eps,delta):
  """"
    conversion of (eps,delta) gurantee to rho-zCDP
    eps : eps in DP 
    delta: delta in DP

    return rho
  """
  func_temp = lambda x: zcdp_to_eps(x,delta) - eps
  root = fsolve(func_temp,x0=0.005)[-1]
  return root
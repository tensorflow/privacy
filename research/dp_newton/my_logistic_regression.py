import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras




class MyLogisticRegression:
  """return a logistic regression problem

  There is a dataset consisting of features (vectors of norm <=1)
  and labels (+1,-1), represented as a numpy array.
  There is also an L2 regularizer.
  """

  def __init__(self,X,y,reg=1e-8):
    """Initialize the data and the regularizer.

    X = n x d numpy array representing features
    y = n x 1 numpy array representing labels
    reg = L2 regularizing coefficient (to ensure solution is finite)

    Data will be rescaled so that ||X[i,:]|| * |y[i]| <= 1 for all i.
    """
    self.reg = float(reg)
    X = np.array(X)
    y = np.array(y)
    assert len(X.shape)==2
    assert len(y.shape)==1
    self.X = X
    self.y = y
    self.n, self.d = X.shape
    assert y.shape[0] == self.n
    signed_data = X * y[:, np.newaxis]
    norm = np.linalg.norm(signed_data, axis=1)
    scale = np.maximum(norm,np.ones_like(norm))
    self.data = (1 / scale[:,None]) * signed_data


  def loss(self,w):
    """Computes the loss represented by this object at w.

    If X,y is the data and reg is the regularizer, then the loss is
    (1/n)sum_i^n log(1+exp(-<w,X[i,:]*y[i]>)) + (reg/2)||w||^2
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, w))))
    reg_loss = 0.5 * self.reg * np.linalg.norm(w)**2
    return data_loss + reg_loss


  def loss_wor(self,w):
    """Computes the loss represented by this object at w without regularizer.

    If X,y is the data and reg is the regularizer, then the loss is
    (1/n)sum_i^n log(1+exp(-<w,X[i,:]*y[i]>))
    """
    data_loss = np.mean(np.log1p(np.exp(-np.dot(self.data, w))))
    return data_loss
    

  def accuracy(self,w):
    """" computes the accuracy of the model gievn by w
    """
    score_pred = np.dot(self.X, w)
    label1_prob = np.where(score_pred >= 0, 1 / (1 + np.exp(-score_pred)), np.exp(score_pred) / (1 + np.exp(score_pred)))
    return np.mean(np.where(label1_prob>=0.5,1,-1) == self.y)


  def grad(self,w):
    """Computes the gradient of the logistic regression at a given point w.

    If X,y is the data and reg is the regularizer, then the gradient is
    (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    coeff_grad = -1/(1+np.exp(np.dot(self.data,w)))
    data_grad = np.mean(self.data * coeff_grad[:, np.newaxis], axis=0)
    return data_grad + self.reg * w


  def grad_wor(self,w):
    """Computes the gradient of the logistic regression at a given point w.

    If X,y is the data and reg is the regularizer, then the gradient is
    (-1/n)sum_i^n X[i,:]*y[i]/(1+exp(<w,X[i,:]*y[i]>)) + reg*w
    """
    coeff_grad = -1/(1+np.exp(np.dot(self.data,w)))
    data_grad = np.mean(self.data * coeff_grad[:, np.newaxis], axis=0)
    return data_grad

  def hess(self,w):
    """Computes the Hessian of the logistic regression at a given point w.

    The Hessian is the matrix of second derivatives.

    If X,y is the data and reg is the regularizer, then the Hessian is
    (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
    where we assume y[i]^2==1.
    """
    a = np.dot(self.data,w)/2
    coeff_hess = 1 / (np.exp(a)+np.exp(-a))**2
    raw_hess = np.dot(self.data.T * coeff_hess, self.data)
    return raw_hess/self.n + self.reg * np.eye(self.d)


  def hess_wor(self,w):
    """Computes the Hessian of the logistic regression at a given point w.

    The Hessian is the matrix of second derivatives.

    If X,y is the data, then the Hessian is
    (1/n)sum_i^n X[i,:]*X[i,:]^T / (cosh(<w,W[i,:]*y[i]>/2)*2)^2
    where we assume y[i]^2==1.
    """
    a = np.dot(self.data,w)/2
    coeff_hess = 1 / (np.exp(a)+np.exp(-a))**2
    raw_hess = np.dot(self.data.T * coeff_hess, self.data)
    return raw_hess/self.n


  def upperbound(self,w):
    """Tightest universal quadratic upper bound on the loss function.
    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor
    This function gives the quadratic term (which replaces the Hessian)
    https://twitter.com/shortstein/status/1557961202256318464
    """
    a = -np.dot(self.data, w)  # vector of y_i<x_i,w> for i in [n]
    # v = 0.5*np.tanh(a/2)/a
    # But avoid 0/0 by special rule
    v = np.divide(0.5*np.tanh(a/2),a,out=(np.ones(a.shape)*0.25),where=(np.abs(a)>1e-9))
    H = np.dot(self.data.T * v,self.data)
    return H/self.n + self.reg * np.eye(self.d)


  def upperbound_wor(self,w):
    """Tightest universal quadratic upper bound on the loss function.
    log(1+exp(x))<=log(1+exp(a))+(x-a)/(1+exp(-a))+(x-a)^2*tanh(a/2)/(4*a)
    Constant and linear terms are just first-order Taylor
    This function gives the quadratic term (which replaces the Hessian)
    https://twitter.com/shortstein/status/1557961202256318464
    """
    a = -np.dot(self.data, w)  # vector of y_i<x_i,w> for i in [n]
    # v = 0.5*np.tanh(a/2)/a
    # But avoid 0/0 by special rule
    v = np.divide(0.5*np.tanh(a/2),a,out=(np.ones(a.shape)*0.25),where=(np.abs(a)>1e-9))
    H = np.dot(self.data.T * v,self.data)
    return H/self.n

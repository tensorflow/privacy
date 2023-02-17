import math
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import sklearn.datasets
import urllib.request
import tensorflow as tf
from tensorflow import keras
import os
from opt_algs import newton


class Mydatasets:
  """Represents datasets we use for testing the algorithms
  """


  def find_optimal_classifier(self, dataset,reg=1e-9):
    """find the optimal weight vector for the logistic regression
        for the problems with real datasets.

    dataset = training dataset
    reg = regularizer
    """
    X,y = dataset
    model_lr = LogisticRegression(max_iter=10000, C=1/reg).fit(X,y)
    w_opt1 = np.concatenate([model_lr.intercept_, np.squeeze(model_lr.coef_)])
    w_opt = newton(dataset,w_opt1)
    print("optimal weight vector norms", np.linalg.norm(w_opt))
    return w_opt


  def adult_dataset(self):
    X = np.load('datasets/adult_processed_x.npy')
    labels = np.load('datasets/adult_processed_y.npy')
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt
  

  def kddcup_dataset(self):
    num_points = 50e3 
    X = np.load('datasets/kddcup99_processed_x.npy',allow_pickle=True)
    labels = np.load('datasets/kddcup99_processed_y.npy',allow_pickle=True)
    X = X.astype(float)
    labels = labels.astype(int)
    selected_samples = np.random.choice(len(X),int(num_points))
    X = X[selected_samples,:]
    labels = labels[selected_samples]
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt
  
  
  def mnist_binary(self):
    """ Download and extract MNIST data
      we also select only two labels for the binary classification task.
    """
    labels=[1,7]
    label0, label1 = int(labels[0]), int(labels[1])
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    nrm = np.linalg.norm(x_train, axis=1)
    x_train = x_train * 1/nrm[:, None]
    y_train = y_train.astype(float)
    indx0 = np.nonzero(y_train == label0)[0]
    indx1 = np.nonzero(y_train == label1)[0]
    y_train[indx0] = -1
    y_train[indx1] = 1
    indx = np.concatenate((indx0,indx1))
    x_train = x_train[indx]
    labels = y_train[indx]
    dataset = x_train, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(x_train)[0],1)), x_train)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt


  def w5a_dataset(self):
    """w5a dataset for logistic regression.
    """
    w5a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w5a"
    data_path = 'datasets/w5a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(w5a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def w7a_dataset(self):
    """w7a dataset for logistic regression.
    """
    w7a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a"
    data_path = 'datasets/w7a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(w7a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt
 
  
  
  
  def w8a_dataset(self):
    """w8a dataset for logistic regression.
    """
    w8a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    data_path = 'datasets/w8a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(w8a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a1a_dataset(self):
    a1a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t"
    data_path = 'datasets/a1a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(a1a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt


  def phishing(self):
    """phishing dataset for logistic regression.
    """
    phishing_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"
    data_path = 'datasets/phishing'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(phishing_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a5a_dataset(self):
    """a5a dataset for logistic regression.
    """
    a5a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a"
    data_path = 'datasets/a5a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(a5a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def covertype_dataset(self):
    X = np.load('datasets/covertype_binary_processed_x.npy')
    labels = np.load('datasets/covertype_binary_processed_y.npy')
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def a9a_dataset(self):
    """a9a dataset for logistic regression.
    """
    a9a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
    data_path = 'datasets/a9a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(a9a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt
  

  def a6a_dataset(self):
    """a6a dataset for logistic regression.
    """
    a6a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a"
    data_path = 'datasets/a6a'
    if not os.path.exists(data_path):
      f = urllib.request.urlretrieve(a6a_url, data_path)
    X, labels = sklearn.datasets.load_svmlight_file(data_path)
    X = X.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    nrm = np.linalg.norm(X, axis=1)
    X = X * 1/nrm[:, None]
    labels = labels.astype(float)
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt

  def synthetic_data(self,n=10000,d=50,cov=None,w=None):
    """Generates a synthetic dataset for logistic regression.

    n = number of samples
    d = dimension
    w = true coefficient vector (optional, default = first standard basis vector)
    cov = covariance of the data (optional, default = identity)

    Features are unit vectors (by default uniformly random).
    Labels are sampled from logistic distribution, so w is the "true" solution.
    """
    mean = np.zeros(d)
    if cov is None:
        cov = np.eye(d)
    X_un = np.random.multivariate_normal(mean, cov, n)
    nrm = np.linalg.norm(X_un, axis=1)
    X = X_un * 1/nrm[:, None]
    if w is None:
        w = np.ones(d)
        w[0]=1
    inner_prod = np.dot(X,w)
    params = np.exp(inner_prod)/(1+np.exp(inner_prod))
    labels = 2*np.random.binomial(1, params)-1
    dataset = X, labels
    w_opt = self.find_optimal_classifier(dataset)
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X)) # adding a dummy dimension for the bias term.
    return X, labels, w_opt





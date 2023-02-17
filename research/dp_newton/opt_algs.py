import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from my_logistic_regression import MyLogisticRegression


class CompareAlgs:
  """Class to run multiple iterative algorithms and compare the results.
  """
  def __init__(self,lr,dataset,optimal_w,iters=10,w0=None,reg=1e-9,pb=None):
    """Initialize the problem.
    
        lr = an instance of MyLogisticRegression 
        dataset = dataset in the format of (features,label)
        optimal_w = optimal minimizer of logistic loss on dataset without privacy
        iters = number of iterations
        w0 = initialization
        reg = regularizer of logistic loss
        pb = {"total": Total privacy budget, "grad_frac": Fraction of privacy budget for gradient vs search direction, "num_iteration": num_iter}

    """
    X,y = dataset
    self.w_opt = optimal_w
    n, d = np.shape(X)
    print("dataset is created: (number of samples, dimension)=" + str(n) + "," + str(d))

    if w0 is None:
      w0_un = np.random.multivariate_normal(np.zeros(d), np.eye(d))
      w0 = w0_un/np.linalg.norm(w0_un)
    self.w0=w0  # initial iterate
    self.iters = iters
    self.pb = pb
    self.lr = lr
    self.params=[]  # List of lists of iterates
    self.names=[]  # List of names
    self.cutoff = 100*np.linalg.norm(self.w_opt)+100*np.linalg.norm(self.w0)+100 # how do you set this value? is it problem-specific?

  def add_algo(self,update_rule,name):
    """Run an iterative update method & add to plot.

    update_rule is a function that takes 4 arguments:
      current iterate
      LogisticRegression problem
      index of current iterate
      total number of iterations
      pb = privacy budget and other info
    """
    baseline=self.lr.loss_wor(self.w_opt)
    w = self.w0
    params = [w]
    for i in range(self.iters):
      w = update_rule(w,self.lr,i,self.iters,self.pb)
      if np.linalg.norm(w)>self.cutoff:
        w=self.w0  # Stop things exploding
        print("Stop Things Exploding!")
      params.append(w)
    self.params.append(params)
    self.names.append(name)
    print()


  def loss_vals(self):
    """output the loss per iteration for different methods
    """
    baseline = self.lr.loss_wor(self.w_opt)
    loss_dict = {}
    for params,name in zip(self.params,self.names):
        losses = [self.lr.loss_wor(w)-baseline for w in params]
        loss_dict[name]=[losses]
    return loss_dict
  
  def accuracy_vals(self):
    """output the accuracy per iteration for different methods
    """
    acc_dict = {}
    for params,name in zip(self.params,self.names):
        acc_vec = [self.lr.accuracy(w) for w in params]
        acc_dict[name]=[acc_vec]
    return acc_dict

  def accuracy_np(self):
    """output the accuracy of the optimal model without privacy
    """
    return self.lr.accuracy(self.w_opt)

  def gradnorm_vals(self):
     """output the gradient norm per iteration for different methods
     """
     gradnorm_dict = {}
     for params,name in zip(self.params,self.names):
         grad_norms = [np.linalg.norm(self.lr.grad_wor(w)) for w in params]
         gradnorm_dict[name]=[grad_norms]
     return gradnorm_dict



def private_newton(w,lr,i,iters,pb):
    """ implementation of private newton method from [ABL21] with non-private backtracking Linesearch

        w = current iterate
        lr = an instance of MyLogisticRegression
        i = the index of current iterate
        iters = total number of iterations
        pb =  privacy budget and other info

        return the next iterate

    """
    total = pb["total"]
    grad_frac = pb["grad_frac"]
    Hess = lr.hess(w)
    rho_grad = grad_frac * total / iters  # divide total privacy budget up
    rho_H = (1-grad_frac) * total / iters
    grad_scale = (1/lr.n)*math.sqrt(0.5/rho_grad)
    grad_noise = np.random.normal(scale=grad_scale,size=lr.d)
    H_scale = (0.25/lr.n)*math.sqrt(0.5/rho_H)
    H_noise = np.random.normal(scale=H_scale,size=(lr.d,lr.d))
    H_noise = (H_noise + H_noise.T)/2
    Hess_noisy = eigenclip(Hess + H_noise)
    dir = np.linalg.solve(Hess_noisy,lr.grad(w)+grad_noise)
    stepsize = backtracking_ls(lr,dir, w)
    return w - stepsize * dir


def eigenclip(A, min_eval=1e-6):
    """ operation of the eigenclip 

        A = symmetric matrix
        min_eval = minimum eigenvalue for clipping

        return the modified matrix
    """
    eval,evec = np.linalg.eigh(A)
    eval = np.maximum(eval,min_eval*np.ones(eval.shape))
    Hclipped = np.dot(evec * eval, evec.T)
    return Hclipped


def gd_priv(w,lr,i,iters,pb):
    """Implementation of DP-GD.

        w = current point
        lr = logistic regression
        i = iteration number
        pb = auxillary information

        output is the next iterate
    """
    inv_lr_gd = 0.25  # learning rate based on the smoothness
    sens = 1/(lr.n*(inv_lr_gd+lr.reg))  # sensitivity
    rho = pb["total"] / iters  # divide total privacy budget up
    noise = np.random.normal(scale=sens/math.sqrt(2*rho),size=lr.d)
    return w - lr.grad(w)/(inv_lr_gd+lr.reg) + noise


def gd_priv_optls(w,lr,i,iters,pb):
    """Implementation of DP-GD with back-tracking line search
        !!! this method is not private. We only use it as a baseline.

        w = current point
        lr = logistic regression
        i = iteration number
        pb = auxillary information

        output is the next iterate
    """
    rho_grad = pb["total"] / iters  # divide total privacy budget up
    grad_scale = (1/lr.n)*math.sqrt(0.5/rho_grad)
    grad_noise = np.random.normal(scale=grad_scale,size=lr.d)
    dir = lr.grad(w)+grad_noise
    stepsize_opt = backtracking_ls(lr,dir, w)
    return w - stepsize_opt * dir


def backtracking_ls(lr,dir, w_0, alpha=0.4, beta=0.95):
    """Implementation of backtracking line search

        lr = logistic regression
        dir = the "noisy" gradient direction
        w_0 = current point
        alpha and beta tradeoff the precision and complexity of the linesearch

        output is an (close to) optimal stepsize
    """
    t = 100
    while lr.loss(w_0 - t*dir) >= lr.loss(w_0) - t*alpha* np.dot(dir, lr.grad(w_0)):
        t = beta * t
        if t <1e-6:
            break
    return t


def newton(dataset,w_init):
    """Implementation of the newton method with linesearch without privacy

        dataset = dataset
        w_init = initialization point
        
        output is the model parameter
    """
    X,y = dataset
    X = np.hstack((np.ones(shape=(np.shape(X)[0],1)), X))
    lr = MyLogisticRegression(X,y,reg=1e-9)
    n, d = np.shape(X)
    w = w_init
    for _ in range(30):
        H = lr.hess(w)
        dir = np.linalg.solve(H,lr.grad(w))
        step_size = backtracking_ls(lr,dir, w)
        w = w - step_size * dir
    if lr.loss_wor(w)<lr.loss_wor(w_init):
        w_out = w
    else:
        w_out = w_init
    return w_out



class DoubleNoiseMech:
    """Our Method: Double Noise Mechanism
    """
    def __init__(self,lr,type_reg='add',hyper_tuning=False,curvature_info='hessian'):
        """ Initializer of the double noise mechanism

            lr = an instance of MyLogisticRegression.
            type_reg = minimum eigenvalue modification type, it can be either 'add' or 'clip'.
            hyper_tuning = True if we want to tune the minimum eigenvalue for modification.
            curvature_info = type of the second-order information, it can be either 'hessian' or 'ub'.
        """
        self.type_reg = type_reg
        self.hyper_tuning = hyper_tuning
        self.curvature_info = curvature_info
        if self.curvature_info == 'hessian':
            self.H = lr.hess_wor
        elif self.curvature_info == 'ub':
            self.H = lr.upperbound_wor

    def find_opt_reg_wop(self,w,lr,noisy_grad,rho_hess):
        """Implementation of fine tuning lambda without privacy.

            w = current point
            lr = logistic regression
            noisy_grad = noisy gradient
            rho_hess = privacy budget for hessian

            output is the optimal minimum eigenvalue
        """
        increase_factor = 1.25 # at each step we increase the clipping by increase_factor
        if self.type_reg == 'add':
            lambda_cur = 5e-6  # starting parameter
        elif self.type_reg == 'clip':
            lambda_cur = 0.25/lr.n + 1e-5 # starting parameter, the denominator has to be greater than zero
        num_noise_sample = 5 # we want to estimate the expected value over the second noise
        grad_norm = np.linalg.norm(noisy_grad)
        H = self.H(w)
        best_loss = 1e6 # a large dummy number
        while lambda_cur <= 0.25:
            H = self.hess_mod(w,lambda_cur)
            if self.type_reg == 'add': # Sensitivity is different for add vs clip
                sens2 = grad_norm * 0.25/(lr.n*lambda_cur**2 + 0.25*lambda_cur)
            elif self.type_reg == 'clip':
                sens2 = grad_norm * 0.25/(lr.n*lambda_cur**2 - 0.25*lambda_cur)
            loss_ = 0
            for _ in range(num_noise_sample):
                noise2 = np.random.normal(scale=sens2 * math.sqrt(0.5/rho_hess), size=lr.d)
                loss_ = loss_ + lr.loss_wor(w - np.linalg.solve(H,noisy_grad) + noise2)
            if loss_ < best_loss:
                best_loss = loss_
                lambda_star = lambda_cur
            lambda_cur = lambda_cur * increase_factor
        return lambda_star

    def update_rule(self,w,lr,i,iters,pb):
        """Implementation of the double noise mechanism update rule

            w = current iterate
            lr = an instance of MyLogisticRegression
            i = the index of current iterate
            iters = total number of iterations
            pb =  privacy budget and other info

            return the next iterate
        """
        total = pb["total"]
        grad_frac = pb["grad_frac"]
        rho1 = grad_frac * total / iters  # divide total privacy budget for gradient
        rho2 = (1-grad_frac) * total / iters  # divide total privacy budget for direction
        sc1 = (1/lr.n) * math.sqrt(0.5/rho1)
        noise1 = np.random.normal(scale=sc1,size=lr.d)
        noisy_grad = lr.grad(w)+noise1
        grad_norm = np.linalg.norm(noisy_grad)
        m = 0.25 # smoothness parameter
        frac_trace = 0.1 #fraction of privacy budget for estimating the trace.
        H = self.H(w)
        if self.hyper_tuning == True:
            min_eval = self.find_opt_reg_wop(w,lr,noisy_grad,rho2)
        elif self.hyper_tuning == False:
            noisy_trace = max(np.trace(H) + np.random.normal(scale=(0.25/lr.n) * math.sqrt(0.5/(frac_trace*rho2))),0)
            min_eval = (noisy_trace/((lr.n)**2 * (1-frac_trace)*rho2))**(1/3) + 8e-4

        H = self.hess_mod(w,min_eval)
        if self.type_reg == 'add': # Sensitivity is different for add vs clip
            sens2 = grad_norm * m/(lr.n * min_eval**2 + m * min_eval)
        elif self.type_reg == 'clip':
            sens2 = grad_norm * m/(lr.n * min_eval**2 - m * min_eval)
        noise2 = np.random.normal(scale=sens2*math.sqrt(0.5/((1-frac_trace)*rho2)),size=lr.d)
        return w - np.linalg.solve(H,noisy_grad) + noise2

    def hess_mod(self,w,min_eval):
      """Implementation of the hessian modification.

            w = current point
            min_eval = minimum eigenvalue for adding or clipping

            output is the modified hessian
      """
      if self.type_reg == 'clip':
          eval,evec = np.linalg.eigh(self.H(w))
          eval = np.maximum(eval,min_eval*np.ones(eval.shape))
          Hclipped = np.dot(evec * eval, evec.T)
          return Hclipped
      elif self.type_reg == 'add':
          return  self.H(w) + min_eval * np.eye(len(self.H(w)))

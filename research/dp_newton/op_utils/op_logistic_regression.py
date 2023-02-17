import numpy as np


class LogisticRegression_OP():
    @staticmethod
    def loss(theta, x, y, lambda_param=None):
        """Loss function for logistic regression with without regularization"""
        exponent = - y * (x.dot(theta))
        return np.sum(np.log(1+np.exp(exponent))) / x.shape[0]

    @staticmethod
    def gradient(theta, x, y, lambda_param=None):
        """
        Gradient function for logistic regression without regularization.
        Based on the above logistic_regression
        """
        exponent = y * (x.dot(theta))
        # stable_val = np.where(exponent<0, 1/(1+np.exp(exponent)) , np.exp(-exponent)/(1+np.exp(-exponent)) )
        gradient_loss = - (np.transpose(x) @ (y / (1+np.exp(exponent)))) / (
            x.shape[0])
        # gradient_loss = - (np.transpose(x) @ (y * stable_val )) / (
            #  x.shape[0])
        # Reshape to handle case where x is csr_matrix
        gradient_loss.reshape(theta.shape)

        return gradient_loss


class LogisticRegressionSinglePoint():
    @staticmethod
    def loss(theta, xi, yi, lambda_param=None):
        exponent = - yi * (xi.dot(theta))
        return np.log(1 + np.exp(exponent))

    @staticmethod
    def gradient(theta, xi, yi, lambda_param=None):

        # Based on page 22 of
        # http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect09.pdf
        exponent = yi * (xi.dot(theta))
        return - (yi*xi) / (1+np.exp(exponent))


class LogisticRegressionRegular():
    @staticmethod
    def loss(theta, x, y, lambda_param):
        regularization = (lambda_param/2) * np.sum(theta*theta)
        return LogisticRegression.loss(theta, x, y) + regularization

    @staticmethod
    def gradient(theta, x, y, lambda_param):
        regularization = lambda_param * theta
        return LogisticRegression.gradient(theta, x, y) + regularization

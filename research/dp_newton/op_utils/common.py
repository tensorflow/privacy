import logging
import numpy as np
from abc import ABCMeta, abstractmethod
import random

LEARNING_RATE_CONSTANT = 0.01
DEFAULT_NUM_ITERS = 100


class Result:
    def __init__(
            self, dataset: str, algorithm: str, epsilon: float, delta: float,
            lambda_param: float, error_rate: float):
        self.dataset = dataset
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.error_rate = error_rate


def test_classifier(x, y, theta):
    """
    Tests the classifier and print the properotions of data
    classified correctly and incorrectly.
    """

    correct, incorrect = compute_classification_counts(x, y, theta)
    n = x.shape[0]

    # This will not work in Python 2
    correct_percentage = (correct/n) * 100
    incorrect_percentage = (incorrect/n) * 100

    print("Correct Percentage: {}".format(correct_percentage))
    print("Incorrect Percentage: {}".format(incorrect_percentage))


def compute_classification_counts(x, y, theta):
    """
    Returns the number of elements classified correctly and the number
    of elements classified incorrectly.

    Meant for binary classifiers
    """

    correct = 0
    incorrect = 0

    n = x.shape[0]
    for i in range(n):
        label = np.sign(x[i] @ theta)
        
        if label == 0:
            label = random.choice([-1, 1])
            
        if label == y[i]:
            correct += 1
        else:
            incorrect += 1

    logging.info("Correct=%d Incorrect=%d n=%d", correct, incorrect, n)

    return correct, incorrect


def compute_multiclass_counts(x, y, thetas):
    """
    Returns the number of elements classified correctly and the number
    of elements classified incorrectly.

    Meant for multiclass classifiers
    """

    correct = 0
    incorrect = 0

    n = x.shape[0]
    for i in range(n):
        max_value = -100000
        index_of_max_value = -1
        for j in range(len(thetas)):
            classification = thetas[j] @ x[i]
            if classification > max_value:
                max_value = classification
                index_of_max_value = j

        if index_of_max_value > -1 and y[i][index_of_max_value] == 1:
            correct += 1
        else:
            incorrect += 1

    return correct, incorrect


def constrain_to_unit_ball(theta):
    """
    Select the closest point contained in a unit ball centered at the origin.
    """

    normalized_theta = np.linalg.norm(theta)
    if normalized_theta > 1:
        theta /= normalized_theta
    return theta


def gen_random_vector_in_unit_ball(size):
    random_vec = np.random.normal(size=size)
    return constrain_to_unit_ball(random_vec)


class Algorithm(metaclass=ABCMeta):
    @abstractmethod
    def run_classification(x, y, epsilon, delta, lambda_param, learning_rate,
                           num_iters):
        return NotImplemented

    @property
    @abstractmethod
    def name(self):
        return NotImplemented

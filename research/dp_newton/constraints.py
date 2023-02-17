import numpy as np

def gen_random_vector_in_unit_ball(size):
    random_vec = np.random.normal(size=size)
    return constrain_to_unit_ball(random_vec)


def constrain_to_unit_ball(theta):
    """
    Select the closest point contained in a unit ball centered at the origin.
    """

    return constrain_l2_norm(theta, 1)


def constrain_l2_norm(theta, constraint):
    """
    Select the closest point contained in a ball centered at the origin.
    """

    normalized_theta = np.linalg.norm(theta, ord=2)
    if normalized_theta > constraint:
        theta /= (normalized_theta / constraint)
    return theta

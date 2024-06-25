from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsc


def exponential(
    scores: np.ndarray,
    sensitivity: float,
    epsilon: float,
    size: int = 1,
    max_fix: bool = True,
    monotonic: bool = False,
    key: int = 0,
) -> np.ndarray:
    """Perform exponential sampling on the scores.

    Args:
        scores (np.ndarray): The scores of the elements in R.
        sensitivity (float): Sensitivity of the score function w.r.t. the private data.
        epsilon (float): pure-differential privacy parameter.
        size (int, optional): Number of independent samplings to perform (e.g. for reporting avg/std of accuracy). Defaults to 1.
        max_fix (bool, optional): Perform a numeric fix by multiplying all probablities with exp(-max_exponent). Defaults to True.
        monotonic (bool, optional): Use lower privacy bound when the score function is monotonic w.r.t. to the private dataset. Defaults to False.
        key (int, optional): Random key for reproducibility. Defaults to 0.

    Returns:
        np.ndarray: array of indice(s) of the sampled element(s).
    """
    if np.isposinf(epsilon):
        max_idx = scores.argmax()
        max_idx = max_idx.repeat(size)
        return max_idx

    sensitivity_factor = 1 if monotonic else 2

    # Substract maximum exponent to avoid overflow
    if max_fix:
        max_exponent = epsilon * scores.max() / (sensitivity_factor * sensitivity)
    else:
        max_exponent = 0
    # Calculate the probability for each element, based on its score
    probabilities = np.exp(
        epsilon * scores / (sensitivity_factor * sensitivity) - max_exponent
    )
    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from R based on the probabilities
    rng = np.random.default_rng(key)
    return rng.choice(len(scores), size, p=probabilities, replace=True)


@jax.jit
def log_binom(n: int, k: int) -> float:
    """Calculate log(n choose k)

    Args:
        n (int): n
        k (int): k

    Returns:
        float: log(n choose k)
    """
    return (
        jsc.special.gammaln(n + 1)
        - jsc.special.gammaln(k + 1)
        - jsc.special.gammaln(n - k + 1)
    )


@partial(
    jax.jit,
    static_argnames=["total_rows", "total_cols"],
)
def exponential_parallel(
    U: jnp.ndarray,
    logm: jnp.ndarray,
    total_rows: int,
    total_cols: int,
    epsilon: float,
    key: int = 42,
) -> jnp.ndarray:
    """Perform parallel exponential sampling of all classes.

    Args:
        U (jnp.ndarray): Scores, shape (total_rows, total_cols) = (n_classes, n_public_samples).
        logm (jnp.ndarray): Log-Counts of the Utilities, shape (total_cols,) = (n_public_samples,).
        total_rows (int): U.shape[0] = n_classes. (needed for jit compilation)
        total_cols (int): U.shape[1] = n_public_samples. (needed for jit compilation)
        epsilon (float): pure-differential privacy parameter.
        key (int, optional): PRNG-initialization. Defaults to 42.

    Returns:
        jnp.ndarray: array of indice(s) of the sampled element(s), shape (total_rows,) = (n_classes,).
    """
    rng = jax.random.key(key)
    choices = (
        jnp.log(jnp.log(1 / jax.random.uniform(rng, (total_rows, total_cols))))
        - logm
        - epsilon * U / 2
    ).argmin(axis=-1)
    return choices


@partial(jax.jit, static_argnames=("total_rows", "total_cols", "k"))
def give_topk_proto_idx(
    U: jnp.ndarray,
    logm: jnp.ndarray,
    k: int,
    total_rows: int,
    total_cols: int,
    epsilon: float,
    key: int = 42,
):
    """Perform the private top-k prototyping. First, perform exponential sampling on the utilities.
    Then, uniformly sample the remaining k-1 prototypes, s.t. their utility is equal or better.

    Args:
        U (jnp.ndarray): Scores, shape (total_rows, total_cols) = (n_classes, n_public_samples).
        logm (jnp.ndarray): Log-Counts of the Utilities, shape (total_cols,) = (n_public_samples,).
        k (int): Number of prototypes per class to sample.
        total_rows (int): U.shape[0] = n_classes. (needed for jit compilation)
        total_cols (int): U.shape[1] = n_public_samples. (needed for jit compilation)
        epsilon (float): pure-differential privacy parameter.
        key (int, optional): PRNG-initialization. Defaults to 42.

    Returns:
        jnp.ndarray: array of indice(s) of the sampled element(s), shape (total_rows, k) = (n_classes, k).
    """
    choices = exponential_parallel(
        U, logm, total_rows, total_cols, epsilon, key
    ).astype(int)

    proto_idx_C = jnp.concatenate(
        [
            jax.lax.select(
                jnp.arange(total_cols)[jnp.newaxis, :].repeat(total_rows, axis=0)
                < choices[:, jnp.newaxis],
                -jax.random.uniform(jax.random.key(key), (total_rows, total_cols)),
                jnp.stack([jnp.zeros((total_cols)) for row in jnp.arange(total_rows)]),
            ).argsort(axis=-1)[:, : k - 1],
            choices[:, jnp.newaxis],
        ],
        axis=1,
    )

    return proto_idx_C

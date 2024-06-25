from functools import partial

import jax
import numpy as np
from jax import numpy as jnp


@jax.jit
def gaussian_tailbound_jit(d, b):
    return (d + 2 * (d * jnp.log(1 / b)) ** 0.5 + 2 * jnp.log(1 / b)) ** 0.5


@partial(jax.jit, static_argnames=("d",))
def multivariate_mean_step_jit(X, c, r, p, n, d, subkey):
    ## Determine a good clipping threshold
    gamma = gaussian_tailbound_jit(d, 0.01)
    clip_thresh = jnp.minimum(
        (r**2 + 2 * r * 3 + gamma**2) ** 0.5, r + gamma
    )  # 3 in place of sqrt(log(2/beta))

    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = jnp.linalg.norm(x, axis=1)

    outside_ball_bool = mag_x > clip_thresh
    x_hat = (x.T / mag_x).T
    X = jnp.where(
        outside_ball_bool[:, jnp.newaxis],
        c + (x_hat * clip_thresh),
        X,
    )

    ## Compute sensitivity
    delta = 2 * clip_thresh / n.astype(float)
    sd = delta / (2 * p) ** 0.5

    ## Add noise calibrated to sensitivity
    Y = sd * jax.random.normal(subkey, (d,))
    c = jnp.sum(X, axis=0) / n.astype(float) + Y
    r = (1 / n.astype(float) + sd**2) ** 0.5 * gaussian_tailbound_jit(d, 0.01)
    return c, r


def multivariate_mean_iterative_jit_inner(i, val, X, Ps, n, d, subkeys):
    c, r = val
    c, r = multivariate_mean_step_jit(X, c, r, Ps[i], n, d, subkeys[i])
    return (c, r)


@partial(jax.jit, static_argnames=("d", "t"))
def multivariate_mean_iterative_jit(X, c, r, t, Ps, n, d, key):
    subkeys = jax.random.split(key, t)
    init_val = c, r
    (c, r) = jax.lax.fori_loop(
        0,
        t,
        partial(
            multivariate_mean_iterative_jit_inner, X=X, Ps=Ps, n=n, d=d, subkeys=subkeys
        ),
        init_val,
    )
    return c


def private_mean_jit(X, Ps, key=jax.random.key(42), r=None, c=None):
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array, but received shape: {}".format(X.shape))
    d = X.shape[1]
    if r is None:
        r = np.sqrt(d) * 0.9
    if c is None:
        c = np.zeros(d)
    t = len(Ps)
    mean = multivariate_mean_iterative_jit(
        X, c=c, r=r, t=t, Ps=Ps, n=X.shape[0], d=d, key=key
    )
    return mean

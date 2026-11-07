from functools import partial

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from omegaconf import DictConfig


def load_dataset(cfg: DictConfig):
  x_train = np.load(cfg.dataset.train_data)
  y_train = np.load(cfg.dataset.train_labels)
  x_test = np.load(cfg.dataset.test_data)
  y_test = np.load(cfg.dataset.test_labels)

  return x_train, y_train, x_test, y_test


def load_public_dataset(cfg: DictConfig):
  x_public = np.load(cfg.dataset.public_data)
  return x_public


def decay(
  cls: int | np.ndarray, max_samples: int, num_classes: int, ratio: float = 10
):
  decay = -np.log(ratio) / num_classes
  return np.round(max_samples * np.exp(decay * cls)).astype(int)


def give_imbalanced_set(x, y, imbalance_ratio: float = 10, seed: int = 42):
  classes = np.unique(y)
  x_classes = [x[y == i] for i in classes]
  rng = np.random.default_rng(seed)
  input_samples_per_class = np.asarray([(y == i).sum() for i in classes])

  output_samples_per_class = decay(
    np.linspace(0, len(classes), len(classes)),
    max_samples=input_samples_per_class.min(),
    num_classes=len(classes),
    ratio=imbalance_ratio,
  )
  rng.shuffle(output_samples_per_class)
  x = np.concatenate(
    [
      x_classes[i][:num_samples]
      for i, num_samples in enumerate(output_samples_per_class)
    ]
  )
  y = np.concatenate(
    [
      np.repeat(i, num_samples)
      for i, num_samples in enumerate(output_samples_per_class)
    ]
  )
  return x, y


def zcdp_of_naive_epsilon(epsilon):
  return epsilon**2 / 2


def exponential_epsilon_of_zcdp(rho):
  return np.sqrt(8 * rho)


@jit
def pairwise_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Calculate 1-cosine_similarity between x and y.

  Args:
      x (jnp.ndarray): x
      y (jnp.ndarray): y

  Returns:
      jnp.ndarray: pairwise distance(s)
  """
  x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
  y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
  return 1 - jnp.dot(x, y.T)


@jit
def scores_single(
  x: jnp.ndarray, y: jnp.ndarray, min_score: float, max_score: float
) -> jnp.ndarray:
  """Score Calculation for a single public sample. The score is calculated as
  the sum of the clipped pairwise distances between the public sample and the
  private samples.

  Args:
      x (jnp.ndarray): private dataset
      y (jnp.ndarray): public sample
      min_score (float): minimum score (in [0,2))
      max_score (float): maximum score (in (min_score, 2])

  Returns:
      jnp.ndarray: Score of the public sample
  """
  return jnp.sum(
    (
      (
        jnp.clip(
          2 - vmap(pairwise_distance, in_axes=(0, None))(x, y),
          min_score,
          max_score,
        )
        - min_score
      )
      / (max_score - min_score)
    ),
    axis=0,
  )


@partial(jit, static_argnames=["batch_size_y"])
def scores_multiple(
  x: jnp.ndarray,
  y: jnp.ndarray,
  min_score: float = 0.0,
  max_score: float = 2.0,
  batch_size_y: int = 5000,
) -> jnp.ndarray:
  """Perform the score calculation batched over the public samples.

  Args:
      x (jnp.ndarray): private dataset
      y (jnp.ndarray): public dataset
      min_score (float, optional): minimum score (in [0,2)). Defaults to 0.0.
      max_score (float, optional): maximum score (in (min_score, 2]). \
        Defaults to 2.0.
      batch_size_y (int, optional): batch size (impacts VRAM usage). \
        Defaults to 5000.

  Returns:
      jnp.ndarray: Scores of all public samples in Y
  """
  return jnp.concatenate(
    [
      vmap(
        partial(scores_single, min_score=min_score, max_score=max_score),
        in_axes=(None, 0),
      )(x, y[i : min(i + batch_size_y, len(y))])
      for i in range(0, len(y), batch_size_y)
    ],
  )

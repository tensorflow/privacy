import warnings
from functools import partial

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from lib import public, utils


@hydra.main(config_path="conf", config_name="public_topk", version_base=None)
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg))

  rho = utils.zcdp_of_naive_epsilon(cfg.epsilon)
  actual_epsilon = utils.exponential_epsilon_of_zcdp(rho)
  print(
    f"Converted settings epsilon {cfg.epsilon} to rho {rho} to exponential \
      epsilon {actual_epsilon}"
  )

  x_train, y_train, x_test, y_test = utils.load_dataset(cfg)
  x_public = utils.load_public_dataset(cfg)
  x_imbalanced, y_imbalanced = utils.give_imbalanced_set(
    x_train, y_train, cfg.imbalance_ratio
  )
  classes = jnp.unique(y_imbalanced)
  try:
    jax.devices("gpu")
  except RuntimeError:
    warnings.warn("No GPU found, falling back to CPU. This will be slow.")
  scores = jnp.stack(
    [
      utils.scores_multiple(
        x_imbalanced[y_imbalanced == target],
        x_public,
        cfg.min_score,
        cfg.max_score,
      )
      for target in classes
    ]
  )
  c_idx = jnp.argsort(scores, axis=1, descending=True)
  if cfg.epsilon < jnp.inf:
    c = jnp.stack([scores[i, c_idx[i]] for i in range(scores.shape[0])])
    u = c - c[:, cfg.k - 1][:, jnp.newaxis]
    with jax.experimental.enable_x64():
      logm = jax.vmap(partial(public.log_binom, k=cfg.k), in_axes=(0))(
        jnp.arange(scores.shape[-1])
      )
    proto_idx_c = public.give_topk_proto_idx(
      u,
      logm,
      cfg.k,
      u.shape[0],
      u.shape[1],
      actual_epsilon,
      cfg.seed,
    )
    proto_idx = jnp.stack(
      [
        c_idx[jnp.arange(c_idx.shape[0]), proto_idx_c[:, k_i]]
        for k_i in range(cfg.k)
      ]
    ).T
  else:
    proto_idx = jnp.stack(
      [c_idx[jnp.arange(c_idx.shape[0]), k_i] for k_i in range(cfg.k)]
    ).T
  public_protos = x_public[proto_idx.flatten()].reshape((*proto_idx.shape, -1))
  dists_test = utils.pairwise_distance(public_protos, x_test)
  test_acc = float((dists_test.argmin(axis=0) == y_test).mean())
  test_acc_per_class = jnp.stack(
    [
      (dists_test[..., y_test == target].argmin(axis=0) == target).mean()
      for target in classes
    ]
  )
  print(f"Test accuracy: {test_acc}")
  print(f"Test accuracy per class: {test_acc_per_class}")


if __name__ == "__main__":
  main()

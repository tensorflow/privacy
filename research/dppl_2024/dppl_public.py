import warnings

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from lib import public, utils


@hydra.main(config_path="conf", config_name="public", version_base=None)
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg))

  rho = utils.zcdp_of_naive_epsilon(cfg.epsilon)
  actual_epsilon = utils.exponential_epsilon_of_zcdp(rho)
  print(
    f"Converted settings epsilon {cfg.epsilon} to rho {rho} to \
      exponential epsilon {actual_epsilon}"
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
  sensitivity = 1.0
  proto_idx_per_class = []
  for target in classes:
    proto_idx_per_class.append(
      public.exponential(
        scores=scores[target],
        sensitivity=sensitivity,
        epsilon=actual_epsilon,
        size=1,
        monotonic=True,
        key=int(cfg.seed + target),
      )
    )
  public_protos = x_public[np.concatenate(proto_idx_per_class)].reshape(
    len(classes), x_public.shape[-1]
  )
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

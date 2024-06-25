import warnings

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from lib import public, utils
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="public", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    rho = utils.zcdp_of_naive_epsilon(cfg.epsilon)
    actual_epsilon = utils.exponential_epsilon_of_zcdp(rho)
    print(
        f"Converted settings epsilon {cfg.epsilon} to rho {rho} to exponential epsilon {actual_epsilon}"
    )

    X_train, Y_train, X_test, Y_test = utils.load_dataset(cfg)
    X_public = utils.load_public_dataset(cfg)
    x_imbalanced, y_imbalanced = utils.give_imbalanced_set(
        X_train, Y_train, cfg.imbalance_ratio
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
                X_public,
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
    public_protos = X_public[np.concatenate(proto_idx_per_class)].reshape(
        len(classes), X_public.shape[-1]
    )
    dists_test = utils.pairwise_distance(public_protos, X_test)
    test_acc = float((dists_test.argmin(axis=0) == Y_test).mean())
    test_acc_per_class = jnp.stack(
        [
            (dists_test[..., Y_test == target].argmin(axis=0) == target).mean()
            for target in classes
        ]
    )
    print(f"Test accuracy: {test_acc}")
    print(f"Test accuracy per class: {test_acc_per_class}")


if __name__ == "__main__":
    main()

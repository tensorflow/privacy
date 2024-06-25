import warnings
from functools import partial

import hydra
import jax
import jax.numpy as jnp
from lib import public, utils
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="public_topk", version_base=None)
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
    C_idx = jnp.argsort(scores, axis=1, descending=True)
    if cfg.epsilon < jnp.inf:
        C = jnp.stack([scores[i, C_idx[i]] for i in range(scores.shape[0])])
        U = C - C[:, cfg.k - 1][:, jnp.newaxis]
        with jax.experimental.enable_x64():
            logm = jax.vmap(partial(public.log_binom, k=cfg.k), in_axes=(0))(
                jnp.arange(scores.shape[-1])
            )
        proto_idx_C = public.give_topk_proto_idx(
            U,
            logm,
            cfg.k,
            U.shape[0],
            U.shape[1],
            actual_epsilon,
            cfg.seed,
        )
        proto_idx = jnp.stack(
            [
                C_idx[jnp.arange(C_idx.shape[0]), proto_idx_C[:, k_i]]
                for k_i in range(cfg.k)
            ]
        ).T
    else:
        proto_idx = jnp.stack(
            [C_idx[jnp.arange(C_idx.shape[0]), k_i] for k_i in range(cfg.k)]
        ).T
    public_protos = X_public[proto_idx.flatten()].reshape((*proto_idx.shape, -1))
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

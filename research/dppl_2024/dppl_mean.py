import flax.linen.pooling as pooling
import hydra
import jax
import jax.numpy as jnp
from lib import coinpress, utils
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="mean", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    X_train, Y_train, X_test, Y_test = utils.load_dataset(cfg)
    X_train = pooling.avg_pool(
        X_train.T, window_shape=(cfg.pool,), strides=(cfg.pool,)
    ).T
    X_test = pooling.avg_pool(X_test.T, window_shape=(cfg.pool,), strides=(cfg.pool,)).T
    x_imbalanced, y_imbalanced = utils.give_imbalanced_set(
        X_train, Y_train, cfg.imbalance_ratio
    )
    classes = jnp.unique(y_imbalanced)
    if cfg.epsilon < jnp.inf:
        rho = utils.zcdp_of_naive_epsilon(cfg.epsilon)
        Ps = jnp.array([5 / 64, 7 / 64, 52 / 64]) * rho
        key = jax.random.key(cfg.seed)
        class_keys = jax.random.split(key, len(classes))
        r = jnp.sqrt(x_imbalanced.shape[1])
        protos = jnp.stack(
            [
                coinpress.private_mean_jit(
                    x_imbalanced[y_imbalanced == i], Ps, key=class_keys[i], r=r
                )
                for i in classes
            ]
        )
    else:
        protos = jnp.stack(
            [x_imbalanced[y_imbalanced == i].mean(axis=0) for i in classes]
        )
    dists_test = utils.pairwise_distance(protos, X_test)
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

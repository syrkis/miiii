# main.py
#   miiiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from functools import partial
import wandb
import miiiii


# functions
def log_run(cfg, metrics, params):
    # long loss and epoch
    n_params = sum([x.size for x in tree_flatten(params)[0]])
    cfg.n_params = n_params
    log_fn = lambda x: {
        "train_loss": x[0],
        "valid_loss": x[1],
        "train_acc": x[2],
        "valid_acc": x[3],
        "train_f1": x[4],
        "valid_f1": x[5],
    }
    with wandb.init(entity="syrkis", project="miiiii", config=cfg):
        for epoch, metric in enumerate(metrics[:-1]):
            wandb.log(log_fn(metric), step=epoch, commit=False)
        wandb.log(log_fn(metrics[-1]), step=cfg.epochs)
        # TODO: log images and model, and maybe more
    wandb.finish()


def main():
    # config and init
    cfg, (rng, key) = miiiii.get_conf(), random.split(random.PRNGKey(0))
    data = miiiii.prime_fn(cfg.n, cfg.base, miiiii.base_n, key)
    params = miiiii.init_fn(key, cfg)

    # train
    apply_fn = miiiii.make_apply_fn(miiiii.vaswani_fn)
    train_fn, state = miiiii.init_train(apply_fn, params, cfg, *data)
    state, metrics = train_fn(cfg.epochs, rng, state)

    # evaluate
    log_run(cfg, metrics, state.params)  # log run


if __name__ == "__main__":
    main()

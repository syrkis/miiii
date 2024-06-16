# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
from functools import partial
import wandb
import miiii


# functions
def log_run(cfg, metrics):
    # long loss and epoch
    log_fn = lambda x: {
        "train_loss": x[0],
        "valid_loss": x[1],
        "train_acc": x[2],
        "valid_acc": x[3],
        "train_f1": x[4],
        "valid_f1": x[5],
    }
    with wandb.init(project="miiii", config=cfg):
        for epoch, metric in enumerate(metrics[:-1]):
            wandb.log(log_fn(metric), step=epoch, commit=False)
        wandb.log(log_fn(metrics[-1]), step=cfg.epochs)
        # TODO: log images and model, and maybe more


def main():
    # config and init
    cfg, (rng, key) = miiii.get_conf(), random.split(random.PRNGKey(0))
    data = miiii.prime_fn(cfg.n, partial(miiii.base_n, cfg.base))
    params = miiii.init_fn(key, cfg)

    # train
    apply_fn = miiii.make_apply_fn(miiii.vaswani_fn)
    train_fn, opt_state = miiii.init_train(apply_fn, params, cfg, *data)
    (params, opt_state), metrics = train_fn(cfg.epochs, rng, (params, opt_state))

    # evaluate
    log_run(cfg, metrics)  # log run


if __name__ == "__main__":
    main()

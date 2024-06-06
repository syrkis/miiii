# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_flatten
import wandb
import src


# functions
def log_run(params, conf, losses):
    train_losses, valid_losses = losses.T  # shape (100,)
    with wandb.init(project="miiii", config=conf):
        for i, (t, v) in enumerate(zip(train_losses, valid_losses)):
            wandb.log({"train_loss": t, "valid_loss": v})


def main():
    # config and init
    conf, (rng, key) = src.get_conf(), random.split(random.PRNGKey(0))
    data = src.prime_fn(conf["n"], partial(src.base_n, conf["base"]))
    params = src.init_fn(key, dict(**conf, len=data[0][0].shape[1]))

    # train
    apply_fn = src.make_apply_fn(src.vaswani_fn)
    loss_fn, train_fn, opt_state = src.init_train(apply_fn, params, conf, *data)
    state, losses = train_fn(params, opt_state, conf["epochs"])

    # evaluate
    src.curve_plot(losses, conf, params)
    train_pred, valid_pred = apply_fn(params, data[0][0]), apply_fn(params, data[1][0])
    src.polar_plot((data[0][1] + train_pred == 0).astype(int), "train")
    # log_run(params, conf, losses)


if __name__ == "__main__":
    main()

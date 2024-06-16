# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_flatten
import wandb
import miiii


# functions
def log_run(params, conf, losses, train_pred, valid_pred, data):
    # long loss and epoch
    log_fn = lambda x: {"train_loss": x[0], "valid_loss": x[1]}
    with wandb.init(project="miiii", config=conf):
        for epoch, loss in enumerate(losses[:-1]):
            wandb.log(log_fn(loss), step=epoch, commit=False)
        wandb.log(log_fn(losses[-1]), step=conf.epochs)
        # TODO: log images and model, and maybe more


def main():
    # config and init
    conf, (rng, key) = miiii.get_conf(), random.split(random.PRNGKey(0))
    data = miiii.prime_fn(conf.n, partial(miiii.base_n, conf.base))
    params = miiii.init_fn(key, conf)

    # train
    apply_fn = miiii.make_apply_fn(miiii.vaswani_fn)
    train_fn, opt_state = miiii.init_train(apply_fn, params, conf, *data)
    (params, opt_state), losses = train_fn(conf.epochs, rng, (params, opt_state))

    # evaluate
    train_pred = miiii.predict(apply_fn, params, data[0][0])
    valid_pred = miiii.predict(apply_fn, params, data[1][0])
    log_run(params, conf, losses, train_pred, valid_pred, data)

    # plot
    miiii.curve_plot(losses, conf, params)
    miiii.polar_plot(data[0][1], train_pred, conf, "train")
    miiii.polar_plot(data[1][1], valid_pred, conf, "valid")


if __name__ == "__main__":
    main()

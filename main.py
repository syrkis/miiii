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
        wandb.log(
            {
                # plot the training and validation loss curves (x-axis is epoch)
                "loss": wandb.Image(src.curve_plot(losses, conf, params)),
                # log the final training and validation losses
                "train_loss": train_losses[-1],
                "valid_loss": valid_losses[-1],
                # log the final training and validation accuracies
            }
        )


def main():
    # config and init
    conf, (rng, key) = src.get_conf(), random.split(random.PRNGKey(0))
    data = src.prime_fn(conf.n, partial(src.base_n, conf.base))
    params = src.init_fn(key, conf)

    # train
    apply_fn = src.make_apply_fn(src.vaswani_fn)
    train_fn, opt_state = src.init_train(apply_fn, params, conf, *data)
    (params, opt_state), losses = train_fn(conf.epochs, rng, (params, opt_state))

    # evaluate
    log_run(params, conf, losses)
    train_pred = src.predict(apply_fn, params, data[0][0])
    valid_pred = src.predict(apply_fn, params, data[1][0])

    # plot
    src.curve_plot(losses, conf, params)
    src.polar_plot(data[0][1], train_pred, conf, "train")
    src.polar_plot(data[1][1], valid_pred, conf, "valid")


if __name__ == "__main__":
    main()

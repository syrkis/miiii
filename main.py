# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rng, key = random.split(random.PRNGKey(ctx.config.seed))
    ds = mi.tasks.task_fn(key, ctx.config)
    state, opt = mi.train.init_fn(rng, ctx.config, ds)
    state, loss = mi.train.train_fn(rng, ctx.config, ds, state, opt)
    # log_fn(ctx, loss)


def log_fn(ctx, loss):
    for i in range(ctx.config.epochs):
        ctx.logger.log_metrics(dict(epoch=i, loss=loss[i % ctx.config.tick, i // ctx.config.tick].item()), "train")


if __name__ == "__main__":
    main()

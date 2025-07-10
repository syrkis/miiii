# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rngs = random.split(random.PRNGKey(ctx.config.seed), 4)
    ds = mi.tasks.task_fn(rngs[0], ctx.config)
    state, opt = mi.train.init_fn(rngs[1], ctx.config, ds)
    loss = mi.train.loss_fn(ds, state.params, rngs[2])
    state, loss = mi.train.train_fn(rngs[3], ctx.config, ds, state, opt)
    log_fn(ctx, loss)


def log_fn(ctx, loss):
    for idx, tick in enumerate(loss):
        for jdx, entry in enumerate(tick):
            ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1] + jdx, loss=entry.item()), "train")


if __name__ == "__main__":
    main()

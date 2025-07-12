# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp


def log_fn(ctx, ds: mi.types.Dataset, state: mi.types.State, loss):
    fns = [mi.plots.plot_params, mi.plots.plot_x, mi.plots.plot_y, mi.plots.plot_log]
    [fn(cfg=ctx.config, ds=ds, params=state.params) for fn in fns]

    for idx, tick in enumerate(loss):
        for jdx, entry in enumerate(tick):
            ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1] + jdx, loss=entry.item()), "train")


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rng = random.PRNGKey(ctx.config.seed)
    ds = mi.tasks.task_fn(rng, ctx.config.p)
    state, opt = mi.train.init_fn(rng, ctx.config, ds)
    state, (loss, scope) = mi.train.train_fn(rng, ctx.config, ds, state, opt)
    # for i in range(ctx.config.epochs):
    # state, loss = mi.train.update_fn(opt, ds, state, rng)
    # if i % ctx.config.tick == 0:
    # print(loss)


if __name__ == "__main__":
    main()

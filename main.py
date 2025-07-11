# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp
import seaborn as sns
import matplotlib.pyplot as plt


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rngs = random.split(random.PRNGKey(ctx.config.seed), 4)
    ds = mi.tasks.task_fn(rngs[0], ctx.config)
    state, opt = mi.train.init_fn(rngs[1], ctx.config, ds)
    logits, z = mi.model.apply(rngs[0], state.params, ds.x)
    loss = mi.train.loss_fn(ds, state.params, rngs[0])
    print(loss)
    # sns.heatmap(ds.mask)
    # plt.show()

    # state, (loss, scope) = mi.train.train_fn(rngs[3], ctx.config, ds, state, opt)
    # print(logits.argmax(-1).deeper)
    # print(ds.y.deeper)
    # print(ds.y[: ctx.config["p"] * 2].T.deeper)
    # print(scope.cce.T.deeper)
    # print(scope.acc.T.deeper)
    # print(ds.task.deeper)
    # log_fn(ctx, ds, state, loss)


def log_fn(ctx, ds: mi.types.Dataset, state: mi.types.State, loss):
    fns = [mi.plots.plot_params, mi.plots.plot_x, mi.plots.plot_y, mi.plots.plot_log]
    [fn(cfg=ctx.config, ds=ds, params=state.params) for fn in fns]

    for idx, tick in enumerate(loss):
        for jdx, entry in enumerate(tick):
            ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1] + jdx, loss=entry.item()), "train")


if __name__ == "__main__":
    main()

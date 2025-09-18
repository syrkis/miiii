# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp


def log_fn(ctx, ds: mi.types.Dataset, state: mi.types.State, loss, scope) -> None:
    for idx, tick in enumerate(loss):
        acc = dict(train_acc=scope.train_acc[idx].tolist(), valid_acc=scope.valid_acc[idx].tolist())
        cce = dict(train_cce=scope.train_cce[idx].tolist(), valid_cce=scope.valid_cce[idx].tolist())
        ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1], **acc, **cce), "scope")
        for jdx, entry in enumerate(tick):
            ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1] + jdx, loss=entry.item()), "train")

    [fn(cfg=ctx.config, ds=ds, params=state.params, scope=scope) for fn in mi.plots.fns]


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    rng = random.PRNGKey(ctx.config.seed)
    ds = mi.tasks.task_fn(rng, ctx.config.p)
    state, opt = mi.train.init_fn(rng, ctx.config, ds)
    state, (loss, scope) = mi.train.train_fn(rng, ctx.config, ds, state, opt)
    log_fn(ctx, ds, state, loss, scope)


# %%
if __name__ == "__main__":
    main()

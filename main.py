# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random, pmap
import miiii as mi
import mlxp
import jax.numpy as jnp
import numpy as np
from functools import partial


def log_fn(ctx, ds: mi.types.Dataset, state: mi.types.State, loss, scope) -> None:
    for mask in ds.primes:
        for idx in range(ctx.config.tick):
            epoch = idx * (ctx.config.epochs // ctx.config.tick)

            acc = dict(train_acc=scope.train_acc[mask, idx].tolist(), valid_acc=scope.valid_acc[mask, idx].tolist())
            cce = dict(train_cce=scope.train_cce[mask, idx].tolist(), valid_cce=scope.valid_cce[mask, idx].tolist())
            ctx.logger.log_metrics(dict(epoch=epoch, **acc, **cce), "scope")

            for jdx in range(ctx.config.epochs // ctx.config.tick):
                ctx.logger.log_metrics(dict(epoch=epoch + jdx, loss=loss[mask, idx, jdx].item()), "train")

            # TODO: log artifacts
            ctx.logger.log_artifacts({"neu": np.array(scope.neu[mask, idx])}, artifact_name="neu.pkl", artifact_type="pickle")
            ctx.logger.log_artifacts({"fft": np.array(scope.fft[mask, idx])}, artifact_name="fft.pkl", artifact_type="pickle")

    # [fn(cfg=ctx.config, ds=ds, params=state.params, scope=scope) for fn in mi.plots.fns]


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    rng = random.PRNGKey(ctx.config.seed)
    ds = mi.tasks.task_fn(rng, ctx.config.p)
    state, opt = mi.train.init_fn(rng, ctx.config, ds)
    mask = (jnp.cumsum(jnp.eye(ds.primes.size), 1) == 1)[::3]
    state, (loss, scope) = pmap(partial(mi.train.train_fn, rng, ctx.config, ds, state, opt))(mask)
    log_fn(ctx, ds, state, loss, scope)


# %%
if __name__ == "__main__":
    main()

# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi
import mlxp

# %% Globals
# arg, cfg = mi.utils.arg_fn(), mi.utils.cfg_fn()

# %% Dataset


# %% Init
# logits, z = mi.model.apply(ds, rng, state.params, ds.x)


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rng, key = random.split(random.PRNGKey(0))
    ds = mi.tasks.task_fn(key, ctx)
    state, opt = mi.train.init_fn(rng, ctx, ds)
    # state, loss = mi.train.train_fn(rng, cfg, arg, ds, state, opt)
    return


if __name__ == "__main__":
    main()

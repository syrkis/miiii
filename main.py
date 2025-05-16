# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi
from jax import random
import jax.numpy as jnp
import numpy as np
from einops import rearrange
import esch

arg = mi.utils.arg_fn()
cfg = mi.utils.cfg_fn()
# study = mi.optim.run_study(arg)

# %% Setup
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg, arg)
state, (scope, metrics, loss) = mi.train.train_fn(rng, cfg, arg, ds)

# scope = scope.transpose(3, 1, 2, 0)[:4]
# esch.grid(scope, path="miiii.svg")
# print(scope.shape)
# mi.utils.log_fn(cfg, arg, ds, state, metrics)

# %%
rearrange(ds.y.train, "(a b) c -> a b c", a=jnp.int32(jnp.sqrt(ds.y.train.shape[0])))
# %%
tmp = rearrange(scope[0][:, 0, 0, :4], "(a b) n -> n a b 1", a=cfg.p)


# %%
arr = np.array(tmp.squeeze()[0])
dwg = esch.init(*arr.shape)
esch.grid_fn(arr / arr.max(), dwg, dwg, shape="square")
esch.save(dwg, "miiii.svg")

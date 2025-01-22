# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi
from jax import random
import jax.numpy as jnp
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
tmp = rearrange(scope[0][:, 0, 0, :4], "(a b) n -> n a b 1", a=cfg.p)
esch.grid(tmp, path="miiii.svg")

# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi

from jax import random


# %% Setup
cfg, arg = mi.utils.cfg_fn(), mi.utils.arg_fn()
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg, arg)
state = mi.utils.State(params=mi.model.init_fn(keys[0], cfg, arg, ds))

# %%
state, (metrics, loss) = mi.train.train_fn(rng, cfg, arg, ds)
mi.utils.log_fn(cfg, arg, ds, state, metrics)

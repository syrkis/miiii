# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random
import miiii as mi

# %% Globals
arg, cfg = mi.utils.arg_fn(), mi.utils.cfg_fn()

# %% Dataset
rng, key = random.split(random.PRNGKey(0))
ds = mi.tasks.task_fn(key, cfg, arg)


# %% Init
state, opt = mi.train.init_fn(rng, cfg, arg, ds)
logits, z = mi.model.apply(ds, rng, state.params, ds.x)
state, loss = mi.train.train_fn(rng, cfg, arg, ds, state, opt)

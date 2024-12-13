# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi

import jax.numpy as jnp
from jax import random, tree, jit, vmap, lax
from functools import partial
from typing import cast
from tqdm import tqdm
import optax


# %% Setup
cfg, arg = mi.utils.cfg_fn(), mi.utils.arg_fn()
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg, arg)
state = mi.utils.State(params=mi.model.init_fn(keys[0], cfg, arg, ds))

# %%


state, metrics = mi.train.train_fn(rng, cfg, arg, ds)

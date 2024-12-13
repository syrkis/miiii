# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi

import jax.numpy as jnp
import yaml
from jax import random, tree


# Load config
cfg = mi.utils.Conf()
args = mi.utils.args_fn()
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, args.task)
state, metrics = mi.train.train(rng, cfg, ds, task)
print(tree.map(jnp.shape, metrics))

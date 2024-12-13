# %% scope.py
#    miiii scope functions
# by: Noah Syrkis

# Imports
from miiii.utils import Conf, Metrics, Split, State
from miiii.tasks import Dataset
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial


def make_scope_fn(cfg, arg, ds):
    x = jnp.concat((ds.x_train, ds.x_val))[ds.udxs]
    def scope_fn(state):
        return None
    return scope_fn

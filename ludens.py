# ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
from jax import random
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from aim import Run, Figure

# %% Initialize
cfg, (rng, key) = mi.utils.load_conf(), random.split(random.PRNGKey(seed := 0))      # test
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, rng)
params = mi.param.init_fn(key, cfg, ds.train.x, ds.train.y)

# %% Training
apply = mi.model.make_apply_fn(mi.model.vaswani_fn)
train, state = mi.train.init_train(apply, params, cfg, ds)
state, metrics = train(cfg.epochs, rng, state)

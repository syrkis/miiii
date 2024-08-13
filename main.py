# %% main.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random, tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
from oeis import A000040 as primes

# %% Exploring and plotting the data
cfg, rng = mi.utils.get_conf(), random.PRNGKey(seed := 0)
rng, key = random.split(rng)
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, key)
params = mi.param.init_fn(key, cfg, ds.train.x, ds.train.y)
apply = mi.model.make_apply_fn(mi.model.vaswani_fn)
train, state = mi.train.init_train(apply, params, cfg, mi.utils.alpha_fn, ds)
state, metrics = train(cfg.epochs, rng, state)

# %% Polar plots
fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
seqs = [range(1, 1024), range(0, 1024, 6), twos_and_fives, primes[1025:2049]]
mi.plots.small_multiples(fnames[:3], seqs[:3], "polar_nats_and_sixes", 1, 3)
mi.plots.polar_plot(seqs[-1], "polar_primes")

# %% Hinton plots
mi.plots.syrkis_plot(metrics["train_loss"], cfg, "Train Focal Loss", ds)

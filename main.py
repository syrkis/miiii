# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random
from oeis import A000040 as primes


# %% Exploring and plotting the data
cfg, rng = mi.utils.cfg_fn(), random.PRNGKey(seed := 0)
rng, key = random.split(rng)
ds = mi.prime.prime_fn(cfg.n, cfg.base, mi.prime.base_ns)

# %% Initialize
params = mi.model.init_fn(key, cfg)
apply = mi.model.apply_fn(cfg)
train, state = mi.train.init_train(apply, params, cfg, ds)

# %% Training
state, metrics = train(cfg.epochs, rng, state)
mi.plots.plot_run(metrics, ds, cfg)
mi.utils.track_metrics(metrics, ds, cfg)

# %% Hinton metrics


# %% Polar plots
fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
seqs = [range(1, 1024), range(0, 1024, 6), twos_and_fives, primes[1025:2049]]
mi.plots.small_multiples(fnames[:3], seqs[:3], "polar_nats_and_sixes", 1, 3)
mi.plots.polar_plot(seqs[-1], "polar_primes")

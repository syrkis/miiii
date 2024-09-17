# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi  # test
from jax import random
from oeis import A000040 as primes


# %% Exploring and plotting the data
cfg = mi.utils.cfg_fn(
    task="prime",
    epochs=100,
    depth=3,
    dropout=0.1,
    l2=0.1,
    heads=4,
    latent_dim=64,
    lr=1e-4,
    n=1024,
    base=23,
)
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.prime.prime_fn(cfg, keys[0])

# %% Initialize
params = mi.model.init_fn(keys[1], cfg)
apply = mi.model.apply_fn(cfg)
train, state = mi.train.init_train(apply, params, cfg, ds)

# %% Training
state, metrics = train(cfg.epochs, rng, state)
mi.plots.plot_run(metrics, ds, cfg)
# mi.utils.track_metrics(metrics, ds, cfg)
# %% Hinton metrics


# %% Polar plots
# fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
# twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
# seqs = [range(1, 1024), range(0, 1024, 2), twos_and_fives, primes[1:300]]
# mi.plots.small_multiples(fnames, seqs, "polar_nats_and_sixes", 1, 4)
# mi.plots.polar_plot(seqs[-1], "polar_primes")

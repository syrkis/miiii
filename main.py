# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi  # test
from jax import random
import optax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
from oeis import A000040 as primes


# %% Exploring and plotting the data 1
cfg = mi.utils.cfg_fn(
    task="prime",
    epochs=2000,
    depth=3,
    dropout=0.2,  # nanda
    l2=1.0,  # nanda
    heads=8,
    latent_dim=256,  # nanda and grokfast
    lr=1e-3,  # like @nanda2023
    # n=12_769,  # 113 ^ 2 @nanda2023 shoutout + 1 (for gpu)
    # base=113,  # 113 is prime
    n=1024,
    base=37,
)

# %% Training
rng, key = random.split(random.PRNGKey(0))
ds = mi.prime.prime_fn(cfg, rng)
state, metrics = mi.train.train(rng, cfg, ds)

# %%
mi.plots.plot_run(metrics, ds, cfg)
# %%
# state, metrics = train(cfg.epochs, rng, state)
# state = train(cfg.epochs, rng, state)
# mi.utils.track_metrics(metrics, ds, cfg)
# %% Hinton metrics


# %% Polar plots
# fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
# twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
# seqs = [range(1, 1024), range(0, 1024, 2), twos_and_fives, primes[1:300]]
# mi.plots.small_multiples(fnames, seqs, "polar_nats_and_sixes", 1, 4)
# mi.plots.polar_plot(seqs[-1], "polar_primes")

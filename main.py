# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi  # test
from jax import random, vmap, lax
import optax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
from oeis import A000040 as primes
from chex import dataclass
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


# %% Training
cfg = mi.utils.cfg_fn(epochs=100, depth=3, lr=1e-3, n=1024, base=2, latent_dim=32)
rng, key = random.split(random.PRNGKey(0))
ds = mi.prime.prime_fn(cfg, rng)
state, metrics = mi.train.train(rng, cfg, ds)
mi.utils.save_params(state, "model.pkl")
# state = mi.utils.load_params("model.pkl")
params = state[0]


# %%
def block_fn(z, param):
    z = (attn := z + mi.model.attn_fn(param.attn, z))
    z = (ffwd := z + mi.model.ffwd_fn(param.ffwd, z))
    return z, (attn, ffwd)


@partial(vmap, in_axes=(None, 0))
def scope_fn(params: mi.kinds.Params, x):
    embeds = mi.model.embed_fn(params.embeddings, x)
    z, acts = lax.scan(block_fn, embeds, params.blocks)
    logits = jnp.mean(z, axis=0) @ params.lm_head
    return embeds, acts, logits


# %%
embeds, (attn_acts, ffwd_acts), logits = scope_fn(params, ds.train.x)
# mi.plots.hinton_activations(attn_acts[0], "./")
# mi.plots.hinton_activations(ffwd_acts[0], "./")

# %%
# mi.plots.plot_run(metrics, ds, cfg)  # type: ignore
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

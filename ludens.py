# %% ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, value_and_grad, grad, tree, lax, nn
import optax

from functools import partial
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# %% Constants
latent_dim = 256
batch_size = 128


# %% Initialize
cfg = mi.utils.cfg_fn(task="prose", latent_dim=latent_dim, batch_size=batch_size)
rng, key1, key2 = random.split(random.PRNGKey(seed := 0), 3)


# %% Training
# apply = mi.model.make_apply_fn(mi.model.vaswani_fn)
# train, state = mi.train.init_train(apply, params, cfg, ds)
# state, metrics = train(cfg.epochs, rng, state)
def loss_fn(params, rng, x, y):
    y_hat = rearrange(apply(params, rng, x), "b s d -> (b s) d")
    y = rearrange(y, "b s -> (b s)")
    loss = optax.softmax_cross_entropy_with_integer_labels(y_hat, y)
    return loss.mean()


# %%
rngs = random.split(random.PRNGKey(0), cfg.batch_size)
apply = jit(vmap(partial(mi.model.apply_fn(cfg), dropout=0.0), in_axes=(None, 0, 0)))
grad_fn = jit(value_and_grad(loss_fn))

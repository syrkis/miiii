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

ficciones, c2i, i2c = mi.prose.prose_fn(key1, cfg)
params = mi.model.init_fn(key2, cfg)  # ds.train.x, ds.train.y)


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

# %%
steps = 2000
pbar = tqdm(zip(range(steps), ficciones), total=steps)
for i, (x, y) in pbar:
    x, y = next(ficciones)
    loss, grads = grad_fn(params, rngs, x, y)
    params = tree.map(lambda p, g: p - cfg.lr * g, params, grads)
    if i % 100 == 0:
        pbar.set_description(f"Loss: {loss.mean()}")


# %% Generate
x = mi.prose.encode_fn("On latitudes this low the sun sets or", c2i)[None, :].repeat(cfg.seq_len, axis=0)

# %%
rng = random.PRNGKey(0)
for i in range(100):
    rng, key = random.split(rng)
    key = random.split(key, x.shape[0])
    y_hat = apply(params, key, x[:, -cfg.seq_len :])[:, -1]
    y = vmap(random.categorical)(key, y_hat)
    x = jnp.concatenate([x, y[:, None]], axis=1)

for i in range(1, 10):
    print(mi.prose.decode_fn(x[0], i2c), end="\n\n\n")
    print(mi.prose.decode_fn(x[1], i2c), end="\n\n\n")
    print(mi.prose.decode_fn(x[2], i2c), end="\n\n\n")
    print(mi.prose.decode_fn(x[3], i2c), end="\n\n\n")
    print(mi.prose.decode_fn(x[4], i2c), end="\n\n\n")

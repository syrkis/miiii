# %% ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
from einops import rearrange
from jax import Array
from jax import random, vmap, jit, value_and_grad, grad, tree, lax, nn
from functools import partial
import jax.numpy as jnp
import jax
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import optax
from aim import Run, Figure
from tqdm import tqdm

# %% Initialize
cfg, (rng, key) = mi.utils.cfg_fn(task="prose"), random.split(random.PRNGKey(seed := 0))  # test
ficciones, c2i, i2c = mi.prose.prose_fn(key, cfg)
params = mi.param.init_fn(key, cfg)  # ds.train.x, ds.train.y)
# %%


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

i, loss = 0, 0
for i, (x, y) in zip(range(1000), ficciones):
    x, y = next(ficciones)
    loss, grads = grad_fn(params, rngs, x, y)
    params = tree.map(lambda p, g: p - cfg.lr * g, params, grads)
print(i, loss)


# %% Generate
x = mi.prose.encode_fn("On latitudes this low the sun sets or", c2i)

rng = random.PRNGKey(0)
for i in range(100):
    rng, key = random.split(rng)
    y_hat = apply(params, key[None, :], x[-cfg.seq_len :][None, :]).squeeze()[-1]
    y = random.categorical(key, y_hat)
    x = jnp.concatenate([x, y[None]], axis=0)
    print(mi.prose.decode_fn(x, i2c))

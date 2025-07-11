# %% model.py
#   miiii model for jax
# by: Noah Syrkis

# Imports
from miiii.types import Params
from miiii.tasks import Dataset
from typing import Tuple
from jax import random, nn, vmap, debug
import jax.numpy as jnp
from jax import Array
from functools import partial


# Constants
initializer: nn.initializers.Initializer = nn.initializers.he_normal()


# Forward
@partial(vmap, in_axes=(None, None, 0))
def apply(key: Array, params: Params, x) -> Tuple[Array, Array]:
    x = embed_fn(params, x)
    x, z = ffwd_fn(params, x)
    logits = jnp.dot(x[-1], params.out_emb)
    return logits, z


def ffwd_fn(params: Params, x: Array) -> Tuple[Array, Array]:
    z = jnp.dot(x, params.w_i)  # z: seq_len x emb_dim
    z = nn.relu(z)  # grokfast relu
    return jnp.dot(z, params.w_o), z  # disable biases as per @nanda2023


def embed_fn(params: Params, x: Array) -> Array:
    tok_emb = jnp.take(params.tok_emb, x, axis=0)
    # print(tok_emb.dtype)
    pos_emb = jnp.take(params.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


# Init
def init_fn(rng: Array, cfg, ds: Dataset) -> Params:
    keys = random.split(rng, 5)
    tok_emb = initializer(keys[0], (cfg.p + 1, cfg.latent_dim))
    pos_emb = initializer(keys[1], (3, cfg.latent_dim))
    out_emb = initializer(keys[2], (ds.primes.size, cfg.latent_dim, cfg.p))  # (task, dim, out)
    w_i = initializer(keys[3], (cfg.latent_dim, cfg.latent_dim * 4))
    w_o = initializer(keys[4], (cfg.latent_dim * 4, cfg.latent_dim))
    return Params(tok_emb=tok_emb, pos_emb=pos_emb, out_emb=out_emb, w_i=w_i, w_o=w_o)


# OTHER #############################################################
# def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
# mask = random.bernoulli(key, 1 - dropout, x.shape)
# return jnp.where(dropout == 0.0, x, mask * x / (1 - dropout))

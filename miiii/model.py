# %% model.py
#   miii model for jax
# by: Noah Syrkis

# Imports
from miiii.types import Params, Feedforward, Embedding
from miiii.tasks import Dataset
import jax
from typing import Tuple
from jax import random, nn, vmap
import jax.numpy as jnp
from jax import Array
from functools import partial


# Constants
initializer = nn.initializers.he_normal()


# Forward
@partial(vmap, in_axes=(None, None, None, 0))
def apply(ds: Dataset, key: Array, params: Params, x) -> Tuple[Array, Array]:
    x = embed_fn(params.embeds, x)
    x, z = ffwd_fn(params.ffwd, x)
    logits = jnp.dot(x[-1], params.unbeds)  # * ds.mask
    return logits, z


def ffwd_fn(w: Feedforward, x: Array) -> Tuple[Array, Array]:
    z = jnp.dot(x, w.w_i)  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # grokfast relu
    return jnp.dot(z, w.w_o), z  # disable biases as per @nanda2023


def embed_fn(w: Embedding, x: Array) -> Array:
    tok_emb = jnp.take(w.tok_emb, x, axis=0)
    pos_emb = jnp.take(w.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


# Init
def init_embed_fn(rng: Array, ctx) -> Embedding:
    keys = random.split(rng, 2)
    tok_emb = initializer(keys[0], (ctx.p + 1, ctx.latent_dim))
    pos_emb = initializer(keys[1], (3, ctx.latent_dim))
    return Embedding(tok_emb=tok_emb, pos_emb=pos_emb)


def init_ffwd_fn(rng: Array, ctx) -> Feedforward:
    w_i = initializer(rng, (ctx.latent_dim, ctx.latent_dim * 4))
    w_o = initializer(rng, (ctx.latent_dim * 4, ctx.latent_dim))
    return Feedforward(w_i=w_i, w_o=w_o)


def init_fn(rng: Array, ctx, ds: Dataset) -> Params:
    keys = random.split(rng, 2 + ctx.depth)
    embeds = init_embed_fn(keys[0], ctx)
    unbeds = initializer(keys[1], (*ds.y.shape[1:], ctx.latent_dim, *(ds.y.shape[-1],)))
    ffwd = init_ffwd_fn(keys[2], ctx)
    return Params(embeds=embeds, unbeds=unbeds, ffwd=ffwd)


# OTHER #############################################################
def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, mask * x / (1 - dropout))

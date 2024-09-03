# model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
from miiiii import kinds
import jax
from jaxtyping import Array
import jax.numpy as jnp
from jax import vmap, random, lax, nn
from einops import rearrange
from functools import partial
from typing import Callable, Tuple


# %% Functions
def predict_fn(apply_fn: Callable, params: kinds.Params, x: Array) -> Array:
    logits = apply_fn(params, random.PRNGKey(0), x, 0.0)
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


# %% Model
# @partial(vmap, in_axes=(None, None, 0, None))
def apply_fn(cfg):  # cfg specifies if the model is causal or not, etc.
    attn = attn_fn(cfg)  # causal or not

    def apply(params: kinds.Params, rng: Array, x: Array, dropout: float = 0.0) -> Array:
        keys = random.split(rng, len(params.blocks))
        z = embed_fn(params, x)  # z: seq_len x emb_dim
        for block in params.blocks:  # use fori_loop maybe
            z = attn(block.head, z)  # use different transformers
            z = ffwd(block.ffwd, z)
            z = lax.cond(dropout > 0.0, z, lambda z: dropout_fn(rng, z, dropout), z, lambda z: z)
        # z = jnp.mean(z, axis=0)  # pool: emb_dim
        logits = z @ params.lm_head  # logits: seq_len x vocab
        return logits  # logits: vocab

    return apply


def attn_fn(cfg: kinds.Conf):  # config specifies if the model is causal or not
    mask = jnp.triu(jnp.full((cfg.seq_len, cfg.seq_len), -jnp.inf), 1)

    def causal_fn(z: Array) -> Array:
        wei = z + mask
        return nn.softmax(wei, axis=-1)

    def attn(params: kinds.Head, x: Array):
        key = x @ params.key
        query = rearrange(x @ params.query, "b T E -> b E T")  # transpose query
        z = (key @ query) / jnp.sqrt(x.shape[-1])  # might be index 0
        z = lax.cond(cfg.causal, z, causal_fn, z, lambda z: z)
        value = x @ params.value
        z = z @ value
        z = rearrange(z, "h t d -> t (h d)")
        return z

    return attn


def dropout_fn(rng: Array, x: Array, rate: float) -> Array:
    rng, key = random.split(rng)
    mask = random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(mask, x / (1 - rate), 0)


def embed_fn(params: kinds.Params, x: Array) -> Array:
    tok_emb = jnp.take(params.tok_emb, x, axis=0)
    pos_emb = jnp.take(params.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def ffwd(params: kinds.FFWD, x: Array) -> Array:
    z = jnp.dot(x, params.w1) + params.b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ params.w2 + params.b2  # disable biases as per @nanda2023
    return z


# def layer_norm(x, gamma, beta, eps=1e-5):
#   mean = jnp.mean(x, axis=-1, keepdims=True)
#   std = jnp.std(x, axis=-1, keepdims=True)
#   return gamma * (x - mean) / (std + eps) + beta

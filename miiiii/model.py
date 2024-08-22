# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
from miiiii import kinds
import jax
import jax.numpy as jnp
from jax import vmap, random, Array
from functools import partial
from einops import rearrange
from typing import Callable, Tuple

# constants
# mask = jnp.triu(jnp.full((x.shape[0], x.shape[0]), -jnp.inf), 1  # for causal masking


def predict_fn(apply_fn: Callable, params: kinds.Params, x: Array) -> Array:
    logits = apply_fn(params, random.PRNGKey(0), x, 0.0)
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


# optional rng
def make_apply_fn(transformer_fn: Callable) -> Callable:
    @partial(vmap, in_axes=(None, None, 0, None))
    def apply_fn(
        params: kinds.Params, rng: Array, x: Array, dropout: float = 0.0
    ) -> Array:
        z = embed_fn(x, params.tok_emb, params.pos_emb)  # z: seq_len x emb_dim
        z, rng = dropout_fn(rng, z, dropout)
        for block in params.blocks:  # use fori_loop maybe
            z = transformer_fn(z, block)  # use different transformers
            z, rng = dropout_fn(rng, z, dropout)
        z = jnp.mean(z, axis=0)  # pool: emb_dim
        logits = z @ params.lm_head  # logits: seq_len x vocab
        return logits.squeeze()  # logits: vocab

    return apply_fn


def dropout_fn(rng: Array, x: Array, rate: float) -> Tuple[Array, Array]:
    rng, key = random.split(rng)
    mask = random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(mask, x / (1 - rate), 0), rng


def embed_fn(x: Array, tok_emb_w: Array, pos_emb_w: Array) -> Array:
    # tok_emb = tok_emb_w[x]  # tok_emb: seq_len x emb_dim
    tok_emb = jnp.take(tok_emb_w, x, axis=0)
    # pos_emb = pos_emb_w[jnp.arange(x.shape[0])]  # pos_emb: seq_len x emb_dim
    pos_idxs = jnp.arange(x.shape[0])
    pos_emb = jnp.take(pos_emb_w, pos_idxs, axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def ffwd_fn(x: Array, params: kinds.FeedForward) -> Array:
    w1, w2, b1, b2 = params.w1, params.w2, params.b1, params.b2
    z = jnp.dot(x, w1) + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2  # disable biases as per @nanda2023
    return z


# function for actually classifying (use sigmoid)
def classify_fn(logits: Array) -> Array:
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


# def layer_norm(x, gamma, beta, eps=1e-5):
#   mean = jnp.mean(x, axis=-1, keepdims=True)
#   std = jnp.std(x, axis=-1, keepdims=True)
#   return gamma * (x - mean) / (std + eps) + beta


######################################
# Vaswani Transformer
######################################

vaswani_ffwd_fn = ffwd_fn


def vaswani_fn(z: Array, block: kinds.Block) -> Array:
    z += vaswani_head_fn(z, block.head)
    z += vaswani_ffwd_fn(z, block.ffwd)
    return z


def vaswani_head_fn(x: Array, params: kinds.Head) -> Array:
    query, key, value, projection = params.query, params.key, params.value, params.proj
    q, k, v = (jnp.dot(x, query), jnp.dot(x, key), jnp.dot(x, value))
    z = q @ rearrange(k, "e t c -> e c t")  # z: embed x seq_len x seq_len
    z /= jnp.sqrt(k.shape[-1])
    wei = jax.nn.softmax(z, axis=-1)
    z = wei @ v  # z: head x seq_len x d_v
    z = rearrange(z, "h t d -> t (h d)")
    z = z @ projection
    return z


######################################
# Hosseini Transformer
######################################


######################################
# He transformer
######################################

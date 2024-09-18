# %% model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
import miiiii as mi

import jax
import optax
from jax import random, value_and_grad, jit, lax, nn, vmap, tree
import jax.numpy as jnp
from jax import Array

from functools import partial
from tqdm import tqdm
from einops import rearrange
from oeis import A000040
from typing import Callable, Optional

# %% Constants #################################################################
initializer = nn.initializers.glorot_uniform()


# %% Model #####################################################################
# attn = attn_fn(cfg)  # causal or not


@partial(vmap, in_axes=(None, None, 0, None))
def apply(params, rng: Array, x: Array, dropout: float) -> Array:
    keys = random.split(rng, len(params.blocks) * 2).reshape(len(params.blocks), 2, 2)
    z = embed_fn(params.embeddings, x)  # z: seq_len x emb_dim
    for key, block in zip(keys, params.blocks):  # use fori_loop maybe
        z = z + attn(block.head, layer_norm(block.ln1, z))
        z = dropout_fn(key[0], z, dropout)
        z = z + ffwd(block.ffwd, layer_norm(block.ln2, z))
        z = dropout_fn(key[1], z, dropout)

    z = layer_norm(params.ln, z)
    z = jnp.mean(z, axis=0)
    logits = z @ params.lm_head  # logits: seq_len x vocab
    return logits  # logits: vocab


def attn(params: mi.kinds.Head, x: Array):
    q, k, v = x @ params.query, x @ params.key, x @ params.value
    z = q @ rearrange(k, "b t c -> b c t")
    z /= jnp.sqrt(params.key.shape[-1])
    z = rearrange(z @ v, "h t d -> t (h d)")
    z = z @ params.proj
    return z


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    return jnp.where(dropout == 0.0, x, x * random.bernoulli(key, 1 - dropout, x.shape) / (1 - dropout))


def embed_fn(params: mi.kinds.Embeddings, x: Array) -> Array:
    tok_emb = jnp.take(params.tok_emb, x, axis=0)
    pos_emb = jnp.take(params.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def ffwd(params: mi.kinds.FFWD, x: Array) -> Array:
    z = jnp.dot(x, params.w1) + params.b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ params.w2 + params.b2  # disable biases as per @nanda2023
    return z


def layer_norm(params: mi.kinds.LayerNorm, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return params.gamma * (x - mean) / (std + 1e-5) + params.beta


# %% Initializers ###########################################################


def init_layer_norm_fn(cfg: mi.kinds.Conf) -> mi.kinds.LayerNorm:
    gamma = jnp.ones(cfg.latent_dim)
    beta = jnp.zeros(cfg.latent_dim)
    return mi.kinds.LayerNorm(gamma=gamma, beta=beta)


def init_head_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Head:
    keys = random.split(rng, 4)
    key = initializer(keys[0], (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads))
    query = initializer(keys[1], (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads))
    value = initializer(keys[2], (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads))
    proj = initializer(keys[3], (cfg.latent_dim, cfg.latent_dim))
    return mi.kinds.Head(query=query, key=key, value=value, proj=proj)


def init_ffwd_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.FFWD:
    keys = random.split(rng)
    w1 = initializer(keys[0], (cfg.latent_dim, cfg.latent_dim * 4))
    w2 = initializer(keys[1], (cfg.latent_dim * 4, cfg.latent_dim))
    b1 = jnp.zeros(cfg.latent_dim * 4)
    b2 = jnp.zeros(cfg.latent_dim)
    return mi.kinds.FFWD(w1=w1, b1=b1, w2=w2, b2=b2)


def init_block_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Block:
    keys = random.split(rng)
    head = init_head_fn(keys[0], cfg)
    ffwd = init_ffwd_fn(keys[1], cfg)
    ln1 = init_layer_norm_fn(cfg)
    ln2 = init_layer_norm_fn(cfg)
    params = mi.kinds.Block(head=head, ffwd=ffwd, ln1=ln1, ln2=ln2)
    return params


def init_embed_fn(rng: Array, cfg: mi.kinds.Conf):
    keys = random.split(rng, 2)
    tok_emb = initializer(keys[0], (cfg.vocab_size, cfg.latent_dim))
    pos_emb = initializer(keys[1], (cfg.seq_len, cfg.latent_dim))
    return mi.kinds.Embeddings(tok_emb=tok_emb, pos_emb=pos_emb)


def init_fn(rng: Array, cfg: mi.kinds.Conf):  # x: Array, y: Array) -> mi.kinds.Params:
    keys = random.split(rng, 3 + cfg.depth)
    params = mi.kinds.Params(
        embeddings=init_embed_fn(keys[0], cfg),
        lm_head=initializer(keys[1], (cfg.latent_dim, y_fn(cfg))),
        blocks=[init_block_fn(key, cfg) for key in keys[3:]],
        ln=init_layer_norm_fn(cfg),  # layer norm for the final layer
    )
    return params


def y_fn(cfg: mi.kinds.Conf) -> int:
    primes = jnp.array(A000040[1 : cfg.n * 2])
    primes = primes[primes < jnp.sqrt(cfg.n)]
    return primes.shape[0] + 1 if cfg.task == "prime" else cfg.vocab_size


# %% Functions #################################################################
def predict_fn(apply_fn: Callable, params: mi.kinds.Params, x: Array) -> Array:
    logits = apply_fn(params, x)
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)

    return jnp.mean(optax.sigmoid_focal_loss(y_hat, y))


def evaluate_splir(ds, params, apply_fn):
    losses = []


# @partial(jit, static_argnums=(0,))
def update(opt, opt_state, grads, params):
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

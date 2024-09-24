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
from typing import Callable, Optional, Tuple
from chex import dataclass

# %% Constants #################################################################
init = nn.initializers.glorot_uniform()


# %% Model #####################################################################
@partial(vmap, in_axes=(None, None, 0, None))
def apply(p: mi.kinds.Params, rng: Array, x: Array, dropout: float) -> Array:
    z = embed_fn(p.embeds, x)
    step_fn = partial(block_fn, dropout=dropout)
    z = lax.scan(step_fn, z, (key_fn(p, rng), p.blocks))[0]
    return jnp.mean(ffwd_fn(p.lm_out, z), axis=0)


def block_fn(z, args, dropout):
    keys, param = args
    z = z + attn_fn(param.attn, z)
    z = dropout_fn(keys[0], z, dropout)
    z = z + ffwd_fn(param.ffwd, z)
    z = dropout_fn(keys[1], z, dropout)
    z = layer_norm(param.norm, z)
    return z, None


def attn_fn(p: mi.kinds.Attention, x: Array):
    q, k, v = x @ p.q, x @ p.k, x @ p.v
    z = q @ rearrange(k, "b t c -> b c t")
    z /= jnp.sqrt(p.k.shape[-1])
    z = nn.softmax(z, axis=-1)
    z = rearrange(z @ v, "h t d -> t (h d)")
    z = z @ p.p
    return z


def ffwd_fn(p: mi.kinds.Feedforward, x: Array) -> Array:
    z = jnp.dot(x, p.w1) + p.b1  # z: seq_len x emb_dim
    z = jax.nn.gelu(z)  # grokfast
    z = z @ p.w2 + p.b2  # disable biases as per @nanda2023
    return z


def embed_fn(p: mi.kinds.Embedding, x: Array) -> Array:
    tok_emb = jnp.take(p.tok_emb, x, axis=0)
    pos_emb = jnp.take(p.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim




def layer_norm(params: mi.kinds.LayerNorm, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return params.gamma * (x - mean) / (std + 1e-5) + params.beta



# %% Initializers ###########################################################
def init_norm_fn(cfg: mi.kinds.Conf) -> mi.kinds.LayerNorm:
    gamma = jnp.ones(cfg.latent_dim)
    beta = jnp.zeros(cfg.latent_dim)
    return mi.kinds.LayerNorm(gamma=gamma, beta=beta)

def init_embed_fn(rng: Array, cfg: mi.kinds.Conf):
    keys = random.split(rng, 2)
    tok_emb = init(keys[0], (cfg.vocab_size, cfg.latent_dim))
    pos_emb = init(keys[1], (cfg.seq_len, cfg.latent_dim))
    return mi.kinds.Embedding(tok_emb=tok_emb, pos_emb=pos_emb)


def init_attn_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Attention:
    keys = random.split(rng, 4)
    shape = (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads)
    q, k, v = init(keys[0], shape), init(keys[1], shape), init(keys[2], shape)
    p = init(keys[3], (cfg.latent_dim, cfg.latent_dim))
    return mi.kinds.Attention(q=q, k=k, v=v, p=p)


def init_ffwd_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Feedforward:
    w1 = init(rng, (cfg.latent_dim, cfg.latent_dim * 4))
    b1 = jnp.zeros(cfg.latent_dim * 4)
    w2 = init(rng, (cfg.latent_dim * 4, cfg.latent_dim))
    b2 = jnp.zeros(cfg.latent_dim)
    return mi.kinds.Feedforward(w1=w1, b1=b1, w2=w2, b2=b2)


def init_block(cfg: mi.kinds.Conf, rng: jnp.ndarray) -> mi.kinds.Block:
    keys = random.split(rng)
    attn = init_attn_fn(keys[0], cfg)
    ffwd = init_ffwd_fn(keys[1], cfg)
    norm = init_norm_fn(cfg)
    return mi.kinds.Block(attn=attn, ffwd=ffwd, norm=norm)

def init_lm_out(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Feedforward:
    keys = random.split(rng, 2)
    w1 = init(keys[0], (cfg.latent_dim, cfg.latent_dim))
    b1 = jnp.zeros(cfg.latent_dim)
    w2 = init(keys[1], (cfg.latent_dim, y_fn(cfg)))
    b2 = jnp.zeros(y_fn(cfg))
    return mi.kinds.Feedforward(w1=w1, b1=b1, w2=w2, b2=b2)

def init_fn(rng: Array, cfg: mi.kinds.Conf):  # x: Array, y: Array) -> mi.kinds.Params:
    keys = random.split(rng, 3 + cfg.depth)
    embeds = init_embed_fn(keys[0], cfg)
    lm_out = init_lm_out(keys[1], cfg)
    blocks = lax.map(partial(init_block, cfg), keys[2:])
    return mi.kinds.Params(embeds=embeds, lm_out=lm_out, blocks=blocks)


# %% Functions #################################################################
def y_fn(cfg: mi.kinds.Conf) -> int:  # infers the number of tasks we are solving
    primes = jnp.array(A000040[1 : cfg.n * 2])
    primes = primes[primes < jnp.sqrt(cfg.n)]
    return primes.shape[0] + 1 if cfg.task == "prime" else cfg.vocab_size


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, x * mask / (1 - dropout))


def key_fn(p, rng):  # split key for dropout
    depth = p.blocks.ffwd.w1.shape[0]
    return random.split(rng, depth * 2).reshape(depth, 2, 2)

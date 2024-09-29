# %% model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
# import miiiii as mi
from miiiii.utils import Conf, digit_fn
from miiiii.scope import Activation

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


@dataclass
class Feedforward:
    w1: Array
    w2: Array


@dataclass
class Attention:
    q: Array
    k: Array
    v: Array
    p: Array


@dataclass
class LayerNorm:
    gamma: Array
    beta: Array


@dataclass
class Block:
    ffwd: Feedforward
    attn: Attention
    # norm: LayerNorm


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array
    # norm: LayerNorm


@dataclass
class Params:
    embeds: Embedding
    blocks: Block
    unbeds: Array  # should be a linear layer ?


# %% Model #####################################################################
def apply_fn(cfg):
    @partial(vmap, in_axes=(None, None, 0, None))
    def apply(p: Params, rng: Array, x: Array, dropout: float) -> Array:
        embed = embed_fn(p.embeds, x)
        step_fn = partial(block_fn, dropout=dropout)
        z = lax.scan(step_fn, embed, (key_fn(p, rng), p.blocks))[0]
        logits = base_n_pos_weigh((z @ p.unbeds), cfg.base)
        return logits.sum(axis=0)

    return apply


def block_fn(z, args, dropout):
    keys, param = args
    z = z + attn_fn(param.attn, z)[0]
    z = dropout_fn(keys[0], z, dropout)
    z = z + ffwd_fn(param.ffwd, z)[0]
    z = dropout_fn(keys[1], z, dropout)
    # z = layer_norm(param.norm, z)
    return z, None


def attn_fn(p: Attention, x: Array):
    q, k, v = x @ p.q, x @ p.k, x @ p.v
    qk = q @ rearrange(k, "b t c -> b c t")
    qk /= jnp.sqrt(p.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    weiv = rearrange(wei @ v, "h t d -> t (h d)")
    weivp = weiv @ p.p
    return weivp, Activation(q=q, k=k, v=v, qk=qk, wei=wei, weiv=weiv)


def ffwd_fn(p: Feedforward, x: Array) -> Tuple[Array, Array]:
    z = jnp.dot(x, p.w1)  # + p.b1  # z: seq_len x emb_dim
    z = jax.nn.tanh(z)  # grokfast
    return z @ p.w2, z  # + p.b2  # disable biases as per @nanda2023


def embed_fn(p: Embedding, x: Array) -> Array:
    tok_emb = jnp.take(p.tok_emb, x, axis=0)
    pos_emb = jnp.take(p.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def layer_norm(params: LayerNorm, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return params.gamma * (x - mean) / (std + 1e-5) + params.beta


# %% Initializers ###########################################################
def init_embed_fn(rng: Array, cfg: Conf):
    keys = random.split(rng, 2)
    tok_emb = init(keys[0], (cfg.vocab_size, cfg.latent_dim))
    pos_emb = init(keys[1], (cfg.seq_len, cfg.latent_dim))
    return Embedding(tok_emb=tok_emb, pos_emb=pos_emb)


def init_attn_fn(rng: Array, cfg: Conf) -> Attention:
    keys = random.split(rng, 4)
    shape = (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads)
    q, k, v = init(keys[0], shape), init(keys[1], shape), init(keys[2], shape)
    p = init(keys[3], (cfg.latent_dim, cfg.latent_dim))
    return Attention(q=q, k=k, v=v, p=p)


def init_ffwd_fn(rng: Array, cfg: Conf) -> Feedforward:
    w1 = init(rng, (cfg.latent_dim, cfg.latent_dim * 4))
    w2 = init(rng, (cfg.latent_dim * 4, cfg.latent_dim))
    return Feedforward(w1=w1, w2=w2)


def init_block(cfg: Conf, rng: jnp.ndarray) -> Block:
    keys = random.split(rng)
    attn = init_attn_fn(keys[0], cfg)
    ffwd = init_ffwd_fn(keys[1], cfg)
    return Block(attn=attn, ffwd=ffwd)  # , norm=norm)


def init_unbeds(rng: Array, cfg: Conf) -> Array:
    keys = random.split(rng, 2)
    unbeds = init(keys[0], (cfg.latent_dim, y_fn(cfg)))
    return unbeds


def init_fn(rng: Array, cfg: Conf):  # x: Array, y: Array) -> mi.types.Params:
    keys = random.split(rng, 2 + cfg.depth)
    embeds = init_embed_fn(keys[0], cfg)
    unbeds = init_unbeds(keys[1], cfg)
    blocks = lax.map(partial(init_block, cfg), keys[2:])
    return Params(embeds=embeds, unbeds=unbeds, blocks=blocks)


# %% Functions #################################################################
def y_fn(cfg: Conf) -> int:  # infers the number of tasks we are solving
    primes = jnp.array(A000040[1 : cfg.n * 2])
    primes = primes[primes < jnp.sqrt(cfg.n)]
    return primes.shape[0] + 1  #  if cfg.task == "prime" else cfg.vocab_size


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, x * mask / (1 - dropout))


def key_fn(p, rng):  # split key for dropout
    depth = p.blocks.ffwd.w1.shape[0]
    return random.split(rng, depth * 2).reshape(depth, 2, 2)


def base_n_pos_weigh(z: Array, base: int) -> Array:
    positions = jnp.arange(z.shape[0])
    weights = jnp.power(base, positions) / jnp.sum(jnp.power(base, positions))  # maybe reverse
    return z * weights[:, None]

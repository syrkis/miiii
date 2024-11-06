# %% model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
# import miiiii as mi
from miiiii.utils import Conf
from dataclasses import field


import jax
from jax import random, lax, nn, vmap
import jax.numpy as jnp
from jax import Array

from functools import partial
from einops import rearrange
from oeis import A000040
from typing import Tuple
from chex import dataclass

# %% Constants #################################################################
init_array = nn.initializers.glorot_uniform()


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


@dataclass
class Activation:
    # q: Array
    # k: Array
    # v: Array
    wei: Array
    ffwd: Array = field(default_factory=lambda: jnp.array([]))  # Use default_factory
    logits: Array = field(default_factory=lambda: jnp.array([]))  # Use default_factory


# %% Model #####################################################################
def apply_fn(cfg: Conf):
    @partial(vmap, in_axes=(None, None, 0, None))  # type: ignore
    def apply(p, rng: Array, x: Array, dropout: float) -> Activation:
        embeds = embed_fn(p.embeds, x)
        step_fn = partial(block_fn, dropout=dropout)
        z, acts = lax.scan(step_fn, embeds, (key_fn(p, rng), p.blocks))
        acts.logits = (z @ p.unbeds).sum(axis=0)
        return acts

    return apply


def block_fn(z, args, dropout):
    keys, param = args
    attn, acts = attn_fn(param.attn, z)
    z = dropout_fn(keys[0], z + attn, dropout)
    ffwd, acts.ffwd = ffwd_fn(param.ffwd, z)[0]
    # print(z.shape)
    # print(ffwd.shape)
    # exit()
    z = dropout_fn(keys[1], z + ffwd, dropout)
    return z, acts


def attn_fn(p, x: Array):
    q, k, v = x @ p.q, x @ p.k, x @ p.v
    qk = q @ rearrange(k, "b t c -> b c t")
    qk /= jnp.sqrt(p.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    return (wei @ v @ p.p).sum(axis=0), Activation(wei=wei)


def ffwd_fn(p: Feedforward, x: Array) -> Tuple[Array, Array]:
    z = jnp.dot(x, p.w1)  # + p.b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # grokfast tanh to things are around 0
    return z @ p.w2, z  # + p.b2  # disable biases as per @nanda2023


def embed_fn(p: Embedding, x: Array) -> Array:
    tok_emb = jnp.take(p.tok_emb, x, axis=0)
    pos_emb = jnp.take(p.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


"""
def layer_norm(params: LayerNorm, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return params.gamma * (x - mean) / (std + 1e-5) + params.beta
"""


# %% Initializers ###########################################################
def init_embed_fn(rng: Array, cfg: Conf):
    keys = random.split(rng, 2)
    tok_emb = init_array(keys[0], (cfg.prime, cfg.latent_dim))  # type: ignore
    pos_emb = init_array(keys[1], (2, cfg.latent_dim))  # type: ignore
    return Embedding(tok_emb=tok_emb, pos_emb=pos_emb)


def init_attn_fn(rng: Array, cfg: Conf) -> Attention:
    keys = random.split(rng, 4)
    shape = (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads)
    q, k, v = map(lambda key: init_array(key, shape), keys[:3])
    p = init_array(keys[3], (cfg.heads, cfg.latent_dim // cfg.heads, cfg.latent_dim))  # type: ignore
    return Attention(q=q, k=k, v=v, p=p)


def init_ffwd_fn(rng: Array, cfg: Conf) -> Feedforward:
    w1 = init_array(rng, (cfg.latent_dim, cfg.latent_dim * 4))  # type: ignore
    w2 = init_array(rng, (cfg.latent_dim * 4, cfg.latent_dim))  # type: ignore
    return Feedforward(w1=w1, w2=w2)


def init_block(cfg: Conf, rng: jnp.ndarray) -> Block:
    keys = random.split(rng)
    attn = init_attn_fn(keys[0], cfg)
    ffwd = init_ffwd_fn(keys[1], cfg)
    return Block(attn=attn, ffwd=ffwd)  # , norm=norm)


def init_fn(rng: Array, cfg: Conf):  # x: Array, y: Array) -> mi.types.Params:
    keys = random.split(rng, 2 + cfg.depth)
    embeds = init_embed_fn(keys[0], cfg)
    unbeds = init_array(keys[1], (cfg.latent_dim, y_fn(cfg)))  # type: ignore
    blocks = lax.map(partial(init_block, cfg), keys[2:])
    return Params(embeds=embeds, unbeds=unbeds, blocks=blocks)


# %% Functions #################################################################
def y_fn(cfg: Conf) -> int:  # infers the number of tasks we are solving
    primes = jnp.array(A000040[1 : cfg.prime**2 * 2])
    primes = primes[primes < jnp.sqrt(cfg.prime**2)]
    tasks = primes.shape[0] + 1  #  if cfg.project == "prime" else cfg.vocab_size
    # TODO: adapt to work with prose
    return tasks if cfg.project == "miiii" else cfg.prime  # if project is nanda we wanna guess the mod


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

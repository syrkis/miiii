# %% model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
from miiii.utils import Conf, Params, Activation, Feedforward, Attention, Embedding
from miiii.tasks import Dataset
import jax
from jax import random, lax, nn, vmap, jit
import jax.numpy as jnp
from jax import Array
from functools import partial
from einops import rearrange

# from einops import rearrange
from oeis import A000040
from typing import Tuple


# %% Constants
initializer = nn.initializers.he_normal()


# %% Forward
def apply_fn(cfg: Conf, ds: Dataset, dropout: float):
    block_fn = make_block_fn(dropout)
    @partial(vmap, in_axes=(None, None, 0))  # type: ignore
    @jit
    def apply(key, params: Params, x) -> Array:
        x = embed_fn(params.embeds, x)
        rngs = random.split(key, cfg.depth)
        x = lax.scan(block_fn, x, (rngs, params.attn, params.ffwd))[0]
        logits = ((x[-1] @ params.unbeds) * ds.mask).squeeze()
        return logits

    return apply

def ffwd_fn(w, x):
    z = jnp.dot(x, w.w_in)  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # grokfast relu
    return jnp.dot(z, w.w_out)  # disable biases as per @nanda2023

def make_block_fn(dropout):
    def block_fn(z, inputs):
        rng, attn_w, ffwd_w = inputs
        keys = random.split(rng, 2)
        attn = attn_fn(attn_w, z)
        z = dropout_fn(keys[0], z + attn, dropout)
        ffwd = ffwd_fn(ffwd_w, z)
        z = dropout_fn(keys[1], z + ffwd, dropout)
        return z, None
    return block_fn


def attn_fn(w, x: Array):
    q, k, v = x @ w.q, x @ w.k, x @ w.v
    qk = jnp.einsum("bth,bsh->bts", q, k) / jnp.sqrt(w.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    return (wei @ v @ w.o).sum(axis=0)  #, Activation(wei=wei)



def embed_fn(w: Embedding, x: Array) -> Array:
    tok_emb = jnp.take(w.tok_emb, x, axis=0)
    pos_emb = jnp.take(w.pos_emb, jnp.arange(x.shape[0]), axis=0)
    return tok_emb + pos_emb  # z: seq_len x emb_dim


# %% Initializers
def init_embed_fn(rng: Array, cfg: Conf):
    keys = random.split(rng, 2)
    tok_emb = initializer(keys[0], (cfg.p + 1, cfg.latent_dim))  # type: ignore
    pos_emb = initializer(keys[1], (3, cfg.latent_dim))  # type: ignore
    return Embedding(tok_emb=tok_emb, pos_emb=pos_emb)


def init_attn_fn(rng: Array, cfg: Conf) -> Attention:
    keys = random.split(rng, 4)
    shape = (cfg.heads, cfg.latent_dim, cfg.latent_dim // cfg.heads)
    q, k, v = map(lambda key: initializer(key, shape), keys[:3])
    o = initializer(keys[3], (cfg.heads, cfg.latent_dim // cfg.heads, cfg.latent_dim))  # type: ignore
    return Attention(q=q, k=k, v=v, o=o)


def init_ffwd_fn(rng: Array, cfg: Conf) -> Feedforward:
    w_in = initializer(rng, (cfg.latent_dim, cfg.latent_dim * 4))  # type: ignore
    w_out = initializer(rng, (cfg.latent_dim * 4, cfg.latent_dim))  # type: ignore
    return Feedforward(w_in=w_in, w_out=w_out)


def init_block(cfg: Conf, rng: jnp.ndarray) -> Tuple[Attention, Feedforward]:
    keys = random.split(rng)
    attn_w = init_attn_fn(keys[0], cfg)
    ffwd_w = init_ffwd_fn(keys[1], cfg)
    return attn_w, ffwd_w


def init_fn(rng: Array, cfg: Conf, arg, ds: Dataset):
    keys = random.split(rng, 2 + cfg.depth)
    embeds = init_embed_fn(keys[0], cfg)
    unbeds = initializer(keys[1], task_size(cfg, arg))
    attn, ffwd = lax.map(partial(init_block, cfg), keys[2:])
    return Params(embeds=embeds, unbeds=unbeds, attn=attn, ffwd=ffwd)


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, mask * x / (1 - dropout))

def task_size(cfg: Conf, arg):
    primes = jnp.array(A000040[1 : cfg.p])
    primes = primes[primes < cfg.p]
    match arg.mods, arg.task:
        case "divisible", "nanda":
            return cfg.latent_dim, 1
        case "remainder", "nanda":
            return cfg.latent_dim, cfg.p
        case "divisible", "miiii":
            return cfg.latent_dim, primes.shape[0]
        case "remainder", "miiii":
            return primes.shape[0], cfg.latent_dim, primes.max()
        case _:
            raise ValueError("Invalid task type or span")

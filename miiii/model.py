# %% model.py
#   miii model for jax
# by: Noah Syrkis

# %% Imports
from miiii.utils import Conf, Params, Activation, Feedforward, Attention, Embedding
from miiii.tasks import Dataset
import jax
from jax import random, lax, nn, vmap
import jax.numpy as jnp
from jax import Array
from functools import partial
from einops import rearrange
from oeis import A000040
from typing import Tuple


# %% Constants
initializer = nn.initializers.he_normal()


# %% Forward
def apply_fn(cfg: Conf):
    @partial(vmap, in_axes=(None, None, 0, None))  # type: ignore
    def apply(params, rng: Array, x: Array, dropout: float) -> Activation:
        embeds = embed_fn(params.embeds, x)
        step_fn = partial(block_fn, dropout=dropout)
        keys = random.split(rng, cfg.depth * 2).reshape(cfg.depth, 2, 2)
        z, acts = lax.scan(step_fn, embeds, (keys, params.attn, params.ffwd))
        acts.logits = (z @ params.unbeds)[-1]  # binary preidction on if x is mul of f.
        return acts

    return apply


def block_fn(z, args, dropout):
    keys, attn_w, ffwd_w = args
    attn, acts = attn_fn(attn_w, z)
    z = dropout_fn(keys[0], z + attn, dropout)
    ffwd, acts.ffwd = ffwd_fn(ffwd_w, z)
    z = dropout_fn(keys[1], z + ffwd, dropout)
    return z, acts


def attn_fn(w, x: Array):
    q, k, v = x @ w.q, x @ w.k, x @ w.v
    qk = q @ rearrange(k, "b t c -> b c t")
    qk /= jnp.sqrt(w.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    return (wei @ v @ w.o).sum(axis=0), Activation(wei=wei)


def ffwd_fn(w: Feedforward, x: Array) -> Tuple[Array, Array]:
    z = jnp.dot(x, w.w_in)  # + w.b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # grokfast relu
    return z @ w.w_out, z  # + w.b2  # disable biases as per @nanda2023


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


def init_fn(rng: Array, cfg: Conf, ds: Dataset):  # -> mi.types.Params:
    keys = random.split(rng, 2 + cfg.depth)
    embeds = init_embed_fn(keys[0], cfg)
    unbeds = initializer(keys[1], n_tasks(cfg))  # type: ignore
    attn, ffwd = lax.map(partial(init_block, cfg), keys[2:])
    return Params(embeds=embeds, unbeds=unbeds, attn=attn, ffwd=ffwd)


# %% Evaluation
def n_tasks(cfg: Conf):  # infers the number of tasks we are solving
    primes = jnp.array(A000040[1 : cfg.p])
    primes = primes[primes < cfg.p]
    shape = (cfg.latent_dim, primes.shape[0]) if cfg.project == "miiii" else (cfg.latent_dim, cfg.p)
    return shape


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, x * mask / (1 - dropout))

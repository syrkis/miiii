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


# %% Model #####################################################################
def apply_fn(cfg, vectorize=True):  # cfg specifies if the model is causal or not, etc.
    attn = attn_fn(cfg)  # causal or not

    def apply(params, rng: Array, x: Array, dropout: float) -> Array:
        keys = random.split(rng, len(params.blocks) * 2).reshape(len(params.blocks), 2, 2)
        z = embed_fn(params.embeddings, x)  # z: seq_len x emb_dim
        for key, block in zip(keys, params.blocks):  # use fori_loop maybe
            z = z + attn(block.head, layer_norm(block.ln1, z))
            z = dropout_fn(key[0], z, cfg, dropout)
            z = z + ffwd(block.ffwd, layer_norm(block.ln2, z))
            z = dropout_fn(key[1], z, cfg, dropout)

        z = layer_norm(params.ln, z)
        z = jnp.mean(z, axis=0)
        logits = z @ params.lm_head  # logits: seq_len x vocab
        return logits  # logits: vocab

    apply = apply if not vectorize else vmap(apply, in_axes=(None, None, 0, None))
    return apply


def attn_fn(cfg: mi.kinds.Conf):  # config specifies if the model is causal or not
    tril = jnp.triu(jnp.full((cfg.seq_len, cfg.seq_len), -jnp.inf), 1)

    def causal_fn(z: Array) -> Array:
        wei = z + tril
        wei = nn.softmax(wei, axis=-1)
        return wei

    def attn(params: mi.kinds.Head, x: Array):
        q, k, v = x @ params.query, x @ params.key, x @ params.value
        z = q @ rearrange(k, "b t c -> b c t")
        z /= jnp.sqrt(params.key.shape[-1])
        # z = lax.cond(cfg.causal, causal_fn, lambda x: x, z)
        # z = dropout_fn(key, z, cfg, dropout)
        z = rearrange(z @ v, "h t d -> t (h d)")
        z = z @ params.proj
        return z

    return attn


def dropout_fn(key: Array, x: Array, cfg: mi.kinds.Conf, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return x * mask / (1 - dropout)


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
    return params.gamma * (x - mean) / (std + 1e-6) + params.beta


# %% Initializers ###########################################################
def init_layer_norm_fn(cfg: mi.kinds.Conf) -> mi.kinds.LayerNorm:
    theta = jnp.sqrt(1 / cfg.latent_dim)
    gamma = jnp.ones(cfg.latent_dim)
    beta = jnp.zeros(cfg.latent_dim)
    return mi.kinds.LayerNorm(gamma=gamma, beta=beta)


def init_head_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Head:
    h, d = cfg.heads, cfg.latent_dim
    keys = random.split(rng, 4)
    theta = jnp.sqrt(1 / d)
    key = random.uniform(keys[0], shape=(h, d, d // h), minval=-theta, maxval=theta)
    query = random.uniform(keys[1], shape=(h, d, d // h), minval=-theta, maxval=theta)
    value = random.uniform(keys[2], shape=(h, d, d // h), minval=-theta, maxval=theta)
    proj = random.uniform(keys[3], shape=(d, d), minval=-theta, maxval=theta)
    return mi.kinds.Head(query=query, key=key, value=value, proj=proj)


def init_ffwd_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.FFWD:
    keys = random.split(rng)
    theta = jnp.sqrt(1 / cfg.latent_dim)
    w1 = random.uniform(keys[0], shape=(cfg.latent_dim, cfg.latent_dim * 4), minval=-theta, maxval=theta)
    w2 = random.uniform(keys[1], shape=(cfg.latent_dim * 4, cfg.latent_dim), minval=-theta, maxval=theta)
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
    tok_emb = random.uniform(
        keys[0],
        shape=(cfg.vocab_size, cfg.latent_dim),
        minval=-jnp.sqrt(1 / cfg.vocab_size),
        maxval=jnp.sqrt(1 / cfg.vocab_size),
    )
    pos_emb = random.uniform(
        keys[1],
        shape=(cfg.seq_len, cfg.latent_dim),
        minval=-jnp.sqrt(1 / cfg.seq_len),
        maxval=jnp.sqrt(1 / cfg.seq_len),
    )
    return mi.kinds.Embeddings(tok_emb=tok_emb, pos_emb=pos_emb)


def init_fn(rng: Array, cfg: mi.kinds.Conf):  # x: Array, y: Array) -> mi.kinds.Params:
    keys = random.split(rng, 3 + cfg.depth)
    params = mi.kinds.Params(
        embeddings=init_embed_fn(keys[0], cfg),
        blocks=[init_block_fn(key, cfg) for key in keys[3:]],
        ln=init_layer_norm_fn(cfg),
        lm_head=random.uniform(
            keys[2],
            shape=(cfg.latent_dim, y_fn(cfg)),
            minval=-jnp.sqrt(1 / cfg.latent_dim),
            maxval=jnp.sqrt(1 / cfg.latent_dim),
        ),
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


def loss_fn(params: mi.kinds.Params, key: Array, x: Array, y: Array, dropout: float = 0.0) -> Array:
    y_hat = apply(params, key, x, dropout)  # should not be hardcoded
    # y_hat = rearrange(y_hat, "b t c -> (b t) c")
    # y = rearrange(y, "b t -> (b t)")
    return jnp.mean(optax.sigmoid_focal_loss(y_hat, y))


def evaluate_splir(ds, params, apply_fn):
    losses = []
    for _ in range(10):
        x, y = next(ds)
        losses.append(loss_fn(params, random.PRNGKey(0), x, y, 0.0))
    return jnp.mean(jnp.array(losses))


@partial(jit, static_argnums=(0,))
def update(opt, opt_state, grads, params):
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state


# %% Test ###################################################################
if __name__ == "__main__":
    rng, key = random.split(random.PRNGKey(0))
    cfg = mi.utils.cfg_fn(latent_dim=256, batch_size=64, seq_len=128, depth=4, heads=8, task="prime")
    # ds_train, ds_test, c2i, i2c, cfg = mi.prose.prose_fn(rng, cfg)
    apply = jit(apply_fn(cfg))
    grad_fn = jit(jax.value_and_grad(loss_fn))
    opt = optax.adamw(0.0001)
    params = init_fn(rng, cfg)

    for i in (pbar := tqdm(range(5000))):
        rng, key = random.split(rng)
    # if i % 100 == 0:
    # train_loss = evaluate_splir(ds_train, params, apply)
    # eval_loss = evaluate_splir(ds_test, params, apply)
    # pbar.set_postfix(train_loss=train_loss, eval_loss=eval_loss)
#
# x = jnp.zeros(
# (
# 1,
# cfg.seq_len,
# )
# ).astype(jnp.int32)
# while True:
# rng, key = random.split(rng)
# y_hat = apply(params, key, x[:, -cfg.seq_len :], 0.0)[:, -1, :]
# pred = random.categorical(key, y_hat).reshape(1, 1)
# x = jnp.concatenate([x, pred], axis=-1)
# print(mi.prose.decode_fn(x[0], i2c))
#

# init.py
#   miiii init
# by: Noah Syrkis

# imports
import miiiii as mi
from jax import random, Array
import jax.numpy as jnp


# constant
theta = 0.01


# init functions
def init_head_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Head:
    h, d = cfg.heads, cfg.emb
    rng, key1, key2, key3 = random.split(rng, 4)

    key = random.uniform(key1, shape=(h, d, d // h), minval=-theta, maxval=theta)
    query = random.uniform(key2, shape=(h, d, d // h), minval=-theta, maxval=theta)
    value = random.uniform(key3, shape=(h, d, d // h), minval=-theta, maxval=theta)
    proj = random.uniform(rng, shape=(h * d // h, d), minval=-theta, maxval=theta)

    return mi.kinds.Head(query=query, key=key, value=value, proj=proj)


def init_ffwd_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.FFWD:
    rng, key1, key2 = random.split(rng, 3)
    w1 = random.uniform(key1, shape=(cfg.emb, cfg.emb), minval=-theta, maxval=theta)
    w2 = random.uniform(key2, shape=(cfg.emb, cfg.emb), minval=-theta, maxval=theta)
    b1 = jnp.zeros(cfg.emb)
    b2 = jnp.zeros(cfg.emb)
    return mi.kinds.FFWD(w1=w1, b1=b1, w2=w2, b2=b2)


def init_block_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Block:
    rng, key1, key2 = random.split(rng, 3)
    params = mi.kinds.Block(head=init_head_fn(key1, cfg), ffwd=init_ffwd_fn(key2, cfg))
    return params


def init_fn(rng: Array, cfg: mi.kinds.Conf, x: Array, y: Array) -> mi.kinds.Params:
    rng, key1, key2, key3 = random.split(rng, 4)
    transformer_keys = random.split(key1, cfg.depth)
    in_d, out_d, emb_d, len_d = (cfg.base, y.shape[1], cfg.emb, x.shape[1])
    params = mi.kinds.Params(
        tok_emb=random.uniform(key1, shape=(in_d, emb_d), minval=-theta, maxval=theta),
        pos_emb=random.uniform(key2, shape=(len_d, emb_d), minval=-theta, maxval=theta),
        blocks=[init_block_fn(transformer_keys[i], cfg) for i in range(cfg.depth)],
        lm_head=random.uniform(key3, shape=(emb_d, out_d), minval=-theta, maxval=theta),
    )
    return params

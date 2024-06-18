# init.py
#   miiii init
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp


# constant
theta = 0.1


# init functions
def init_head_fn(rng, cfg):
    h, d = cfg.heads, cfg.emb
    rng, key1, key2, key3 = random.split(rng, 4)

    key = random.uniform(key1, shape=(h, d, d // h), minval=-theta, maxval=theta)
    query = random.uniform(key2, shape=(h, d, d // h), minval=-theta, maxval=theta)
    value = random.uniform(key3, shape=(h, d, d // h), minval=-theta, maxval=theta)
    projection = random.uniform(rng, shape=(h * d // h, d), minval=-theta, maxval=theta)

    return (query, key, value, projection)


def init_ffwd_fn(rng, cfg):
    rng, key1, key2 = random.split(rng, 3)
    emb_dim = cfg.emb
    shape = (emb_dim, 4 * emb_dim)
    w1 = random.uniform(key1, shape=shape, minval=-theta, maxval=theta)
    w2 = random.uniform(key2, shape=shape[::-1], minval=-theta, maxval=theta)
    b1 = jnp.zeros(emb_dim * 4)
    b2 = jnp.zeros(emb_dim)
    return w1, w2, b1, b2


def init_block_fn(rng, cfg):
    rng, key1, key2 = random.split(rng, 3)
    params = {
        "head": init_head_fn(key1, cfg),
        "ffwd": init_ffwd_fn(key2, cfg),
    }
    return params


def init_fn(rng, cfg):
    rng, key1, key2, key3 = random.split(rng, 4)
    transformer_keys = random.split(key1, cfg.depth)
    in_d, out_d, emb_d, len_d = cfg.base, 1, cfg.emb, cfg.digits
    params = dict(
        tok_emb=random.uniform(key1, shape=(in_d, emb_d), minval=-theta, maxval=theta),
        pos_emb=random.uniform(key2, shape=(len_d, emb_d), minval=-theta, maxval=theta),
        blocks=[init_block_fn(transformer_keys[i], cfg) for i in range(cfg.depth)],
        lm_head=random.uniform(key3, shape=(emb_d, out_d), minval=-theta, maxval=theta),
    )
    return params


if __name__ == "__main__":
    from utils import load_cfg

    cfg = load_cfg()
    rng = random.PRNGKey(0)
    params = init_fn(rng, cfg)
    print(params)
    print(len(params["blocks"]))
    print(params["blocks"][0]["head"][0].shape)
    print(params["blocks"][0]["ffwd"][0].shape)
    print(params["tok_emb"].shape)
    print(params["pos_emb"].shape)
    print(params["lm_head"].shape)

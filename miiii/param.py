# init.py
#   miiii init
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp


# init functions
def init_head_fn(rng, cfg):
    h, d = cfg.heads, cfg.emb
    rng, key1, key2, key3 = random.split(rng, 4)

    key = random.uniform(key1, shape=(h, d, d // h), minval=-0.1, maxval=0.1)
    query = random.uniform(key2, shape=(h, d, d // h), minval=-0.1, maxval=0.1)
    value = random.uniform(key3, shape=(h, d, d // h), minval=-0.1, maxval=0.1)
    projection = random.uniform(rng, shape=(h * d // h, d), minval=-0.1, maxval=0.1)
    gamma, beta = jnp.ones((d)), jnp.zeros((d))

    return (query, key, value, projection, gamma, beta)


def init_ffwd_fn(rng, cfg):
    rng, key1, key2 = random.split(rng, 3)
    emb_dim = cfg.emb
    gamma, beta = jnp.ones((emb_dim)), jnp.zeros((emb_dim))
    params = (  # multiply by 4 for cos thats what people do
        random.uniform(key1, shape=(emb_dim, 4 * emb_dim), minval=-0.1, maxval=0.1),
        jnp.zeros((4 * emb_dim)),
        random.uniform(key2, shape=(4 * emb_dim, emb_dim), minval=-0.1, maxval=0.1),
        jnp.zeros((emb_dim)),
        gamma,
        beta,
    )
    return params


def init_block_fn(rng, cfg):
    rng, key1, key2 = random.split(rng, 3)
    params = {
        "head": init_head_fn(key1, cfg),
        "ffwd": init_ffwd_fn(key2, cfg),
    }
    return params


def init_fn(rng, cfg):
    rng, key1, key2, key3 = random.split(rng, 4)
    in_d, out_d, emb_d, len_d = cfg.base, 1, cfg.emb, cfg.digits
    params = dict(
        tok_emb=random.uniform(key1, shape=(in_d, emb_d), minval=-0.1, maxval=0.1),
        pos_emb=random.uniform(key2, shape=(len_d, emb_d), minval=-0.1, maxval=0.1),
        blocks=[init_block_fn(key1, cfg) for _ in range(cfg.depth)],
        lm_head=random.uniform(key3, shape=(emb_d, out_d), minval=-0.1, maxval=0.1),
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

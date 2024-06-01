# init.py
#   miiii init
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp


# init functions
def init_head_fn(rng, conf):
    h, d, scale = conf["n_heads"], conf["emb"], conf["scale"]
    head_size = conf["emb"] // conf["n_heads"]

    rng, key1, key2, key3 = random.split(rng, 4)
    key = random.normal(key1, shape=(h, d, head_size)) * scale
    query = random.normal(key2, shape=(h, d, head_size)) * scale
    value = random.normal(key3, shape=(h, d, head_size)) * scale
    return (query, key, value)


def init_ffwd_fn(rng, conf):
    rng, key1, key2 = random.split(rng, 3)
    emb_dim, scale = conf["emb"], conf["scale"]
    params = (  # multiply by 4 for cos thats what people do
        random.uniform(key1, shape=(emb_dim, emb_dim), minval=-0.1, maxval=0.1),
        jnp.zeros((emb_dim)),
        random.uniform(key2, shape=(emb_dim, emb_dim), minval=-0.1, maxval=0.1),
        jnp.zeros((emb_dim)),
    )
    return params


def init_block_fn(rng, conf):
    rng, key1, key2 = random.split(rng, 3)
    params = {
        "head": init_head_fn(key1, conf),
        "ffwd": init_ffwd_fn(key2, conf),
    }
    return params


def init_fn(rng, conf):
    rng, key1, key2, key3 = random.split(rng, 4)
    in_d, out_d, emb_d, len_d = conf["in_d"], conf["out_d"], conf["emb"], conf["len"]
    params = dict(
        tok_emb=random.uniform(key1, shape=(in_d, emb_d), minval=-0.1, maxval=0.1),
        pos_emb=random.uniform(key2, shape=(len_d, emb_d), minval=-0.1, maxval=0.1),
        blocks=[init_block_fn(key1, conf) for _ in range(conf["n_layers"])],
        lm_head=random.uniform(key3, shape=(emb_d, out_d), minval=-0.1, maxval=0.1),
    )
    return params


if __name__ == "__main__":
    from utils import load_conf

    conf = load_conf()
    rng = random.PRNGKey(0)
    params = init_fn(rng, conf)
    print(params)
    print(len(params["blocks"]))
    print(params["blocks"][0]["head"][0].shape)
    print(params["blocks"][0]["ffwd"][0].shape)
    print(params["tok_emb"].shape)
    print(params["pos_emb"].shape)
    print(params["lm_head"].shape)

# init.py
#   miiii init
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp


# init functions
def init_head_fn(rng, conf):
    rngs = random.split(rng, conf["n_heads"])
    shape = (conf["n_heads"], conf["emb_dim"], conf["emb_dim"] // conf["n_heads"])
    keys = random.normal(rngs[0], shape=shape) * conf["scale"]
    values = jnp.zeros_like(keys)
    alpha, beta = jnp.array(1.0), jnp.array(0.0)
    return (keys, values, alpha, beta)


def init_ffwd_fn(rng, conf):
    rng, key1, key2 = random.split(rng, 3)
    emb_dim, scale = conf["emb_dim"], conf["scale"]
    params = (  # multiply by for cos thats what people do
        random.normal(key1, shape=(emb_dim, emb_dim * 4)) * scale,
        jnp.zeros((emb_dim * 4)).astype(jnp.float32),
        random.normal(key2, shape=(4 * emb_dim, emb_dim)) * scale,
        jnp.zeros((emb_dim)).astype(jnp.float32),
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
    base, emb_dim, block_size = conf["base"], conf["emb_dim"], conf["block_size"]
    base = base if "vocab" not in conf else conf["vocab"]
    params = {
        "tok_emb": random.normal(key1, shape=(base, emb_dim)) * conf["scale"],
        "pos_emb": random.normal(key2, shape=(block_size, emb_dim)) * conf["scale"],
        "lm_head": random.normal(key3, shape=(emb_dim, conf["vocab"])) * conf["scale"],
        "blocks": [init_block_fn(key1, conf) for _ in range(conf["n_layers"])],
    }
    return params


if __name__ == "__main__":
    from utils import load_conf

    conf = load_conf(128)
    rng = random.PRNGKey(0)
    params = init_fn(rng, conf)
    print(params)
    print(len(params["blocks"]))
    print(params["blocks"][0]["head"][0].shape)
    print(params["blocks"][0]["mlp"][0].shape)
    print(params["tok_emb"].shape)
    print(params["pos_emb"].shape)
    print(params["lm_head"].shape)

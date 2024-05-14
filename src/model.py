# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random
from jax.tree_util import tree_flatten, tree_unflatten

from functools import partial
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple, Union


# forward functions
@partial(vmap, in_axes=(None, 0))
def apply_fn(params, x):
    x = embed_fn(params, x)  # T x d
    activations = [x]
    for block in params["blocks"]:
        x = head_fn(x, *block["head"]) + mlp_fn(x, *block["mlp"])
        activations.append(x)
    logits = x @ params["lm_head"]
    return jax.nn.sigmoid(logits[-1])


def head_fn(x, w_key, w_query, alpha, beta):
    z = x @ w_query @ w_key.transpose(0, 2, 1) @ x.transpose(1, 0)  # H x T x T
    z /= jnp.sqrt(w_key.shape[1])
    z = jax.nn.softmax(z, axis=-1)
    z = alpha * z + beta * jnp.eye(x.shape[0])
    z = (z @ x).mean(axis=0)
    return z


def mlp_fn(x, dense1, bias1, dense2, bias2):
    z = x @ dense1 + bias1
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ dense2 + bias2
    return z


def embed_fn(params, x):
    z = params["tok_emb"][x]  # T x d
    z += params["pos_emb"][jnp.arange(x.shape[0])]  # pos embeddings
    return z


# init functions
def init_head_fn(rng, conf):
    rngs = random.split(rng, conf["n_heads"])
    shape = (conf["n_heads"], conf["emb_dim"], conf["emb_dim"] // conf["n_heads"])
    keys = random.normal(rngs[0], shape=shape) * conf["scale"]
    values = jnp.zeros_like(keys).astype(jnp.float32)
    alpha, beta = jnp.array(1.0), jnp.array(0.0)
    return (keys, values, alpha, beta)


def init_mlp_fn(rng, conf):
    rng, key1, key2 = random.split(rng, 3)
    emb_dim, scale = conf["emb_dim"], conf["scale"]
    params = (
        random.normal(key1, shape=(emb_dim, emb_dim)) * scale,
        jnp.zeros((emb_dim)).astype(jnp.float32),
        random.normal(key2, shape=(emb_dim, emb_dim)) * scale,
        jnp.zeros((emb_dim)).astype(jnp.float32),
    )
    return params


def init_block_fn(rng, conf):
    rng, key1, key2 = random.split(rng, 3)
    params = {
        "head": init_head_fn(key1, conf),
        "mlp": init_mlp_fn(key2, conf),
    }
    return params


def init_fn(rng, conf):
    rng, key1, key2, key3 = random.split(rng, 4)
    base, emb_dim, block_size = conf["base"], conf["emb_dim"], conf["block_size"]
    params = {
        "tok_emb": random.normal(key1, shape=(base, emb_dim)) * conf["scale"],
        "pos_emb": random.normal(key2, shape=(block_size, emb_dim)) * conf["scale"],
        "lm_head": random.normal(key3, shape=(emb_dim, 1)) * conf["scale"],
        "blocks": [init_block_fn(key1, conf) for _ in range(conf["n_layers"])],
    }
    return params


def main():
    rng = random.PRNGKey(0)
    conf = conf_fn()
    x, y = data_fn(conf)
    params = init_fn(rng, conf)
    logits = apply_fn(params, x)
    print(logits.shape)


if __name__ == "__main__":
    from utils import conf_fn
    from data import data_fn

    main()

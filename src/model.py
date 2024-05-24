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
    return jax.nn.softmax(logits[-1])


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


if __name__ == "__main__":
    from utils import load_conf
    from param import init_fn
    from data import conrad_fn

    rng, data_key, param_key = random.split(random.PRNGKey(0), 3)
    data, c2i, i2c = conrad_fn(data_key, 128)
    params = init_fn(param_key, load_conf())
    pred = apply_fn(params, data)
    print(pred.shape)

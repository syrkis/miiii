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


# CORRECT
@partial(vmap, in_axes=(None, 0))
def apply_fn(params, x):  # x: seq_len
    z = embed_fn(x, params["tok_emb"], params["pos_emb"])  # z: seq_len x emb_dim
    # for block in params["blocks"]:
    #     z += head_fn(z, *block["head"])  # <- this the hardcore stuff
    #     z += ffwd_fn(z, *block["ffwd"])
    logits = z @ params["lm_head"]  # logits: seq_len x vocab
    return jax.nn.log_softmax(logits)


# ((x @ w_key) @ w_query.T @ x.T / sqrt(d)  * beta + alpha * I) @ X
def head_fn(x, w_key, w_query, alpha, beta):  # x: seq_len x emb_dim
    # copy x
    z = x  # z: seq_len x emb_dim

    # mulitply by w_query (probably CORRECT)
    z @= w_query  # z: heads x seq_len x emb_dim // heads

    # multiply by w_key (also probably CORRECT, but strange with head dim)
    z @= w_key.transpose(0, 2, 1)  # z: heads x seq_len x emb_dim

    # multiply by x transpose (probably CORRECT)
    z @= x.T  # z: heads x seq_len x seq_len

    # scale by sqrt (probably CORRECT)
    z /= jnp.sqrt(w_key.shape[1])

    # scale by beta which is 0 initially (probably CORRECT)
    z *= beta

    # add identity matrix scaled by alpha which is 1 initually (probably CORRECT)
    z += alpha * jnp.eye(x.shape[0])

    # multiply by x (probably CORRECT)
    z @= x  # z: heads x seq_len x emb_dim

    # concat  # TODO: this is wrong, cos it makes the dim 4 times bigger
    z = rearrange(z, "h s d -> s (h d)")

    return z


# CORRECT
def ffwd_fn(x, w1, b1, w2, b2):
    z = x @ w1 + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2
    return z


# CORRECT
def embed_fn(x, tok_emb, pos_emb):  # x: seq_len
    z = tok_emb[x]  # z: seq_len x emb_dim
    z += pos_emb[jnp.arange(x.shape[0])]
    return z


def generate_fn(params, x, rng, max_len=None, decode=None):
    while x.shape[1] < max_len:
        rng, key = random.split(rng)
        pred = random.categorical(key, apply_fn(params, x)[:, -1])
        x = jnp.concatenate([x, pred[:, None]], axis=1)
        None if decode is None else print(decode(x[0]))
    return x


if __name__ == "__main__":
    from utils import load_conf
    from param import init_fn
    from datum import text_fn

    rng, data_key, param_key = random.split(random.PRNGKey(0), 3)
    data, encode, decode, vocab = text_fn(data_key, 8, 4)
    params = init_fn(param_key, load_conf(len(vocab)))
    x, y = next(data)
    x = generate_fn(params, x, rng, 100)

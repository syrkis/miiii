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
from tqdm import tqdm


# CORRECT
@partial(vmap, in_axes=(None, 0))
def apply_fn(params, x):  # x: seq_len
    z = embed_fn(x, params["tok_emb"], params["pos_emb"])  # z: seq_len x emb_dim
    # activations = [z]  # [h, z, h, ..., z, h]  like patern
    for block in params["blocks"]:
        z += head_fn(z, *block["head"])  # <- this the hardcore stuff
        z += ffwd_fn(z, *block["ffwd"])
        # activations += [h, z]
    logit = (z @ params["lm_head"]).squeeze()[-1]  # logits: seq_len x vocab
    return jax.nn.sigmoid(logit)


def head_fn(x, w_key, w_query, alpha, beta):  # x: seq_len x emb_dim
    z = x  # z: seq_len x emb_dim
    z @= w_query  # z: heads x seq_len x emb_dim // heads
    z @= w_key.transpose(0, 2, 1)  # z: heads x seq_len x emb_dim
    z @= x.T  # z: heads x seq_len x seq_len
    z /= jnp.sqrt(w_key.shape[1])
    z *= beta
    z += alpha * jnp.eye(x.shape[0])
    z @= x  # z: heads x seq_len x emb_dim
    return z.mean(axis=0)  # TODO: this is wrong!


# CORRECT
def ffwd_fn(x, w1, b1, w2, b2, alpha):
    z = x @ w1 + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2
    return alpha * z


# CORRECT
def embed_fn(x, tok_emb, pos_emb):  # x: seq_len
    z = tok_emb[x]  # z: seq_len x emb_dim
    z += pos_emb[jnp.arange(x.shape[0])]
    return z


# @partial(jit, static_argnums=(3,))
def generate_fn(params, x, rng, length=42):
    for _ in tqdm(range(length)):
        rng, key = random.split(rng)
        prob = apply_fn(params, x)[:, -1]
        pred = random.categorical(key, prob)
        x = jnp.concatenate([x, pred[:, None]], axis=1)
    return x


if __name__ == "__main__":
    from utils import load_conf
    from param import init_fn
    from datum import data_fn
    from numbs import base_n
    from oeis import oeis

    # rng, data_key, param_key = random.split(random.PRNGKey(0), 3)
    rng, key = random.split(random.PRNGKey(0))
    ns = partial(base_n, n=2)
    x, y = data_fn(oeis["A000040"], 2**10, ns)
    params = init_fn(key, load_conf(1))
    pred = apply_fn(params, x)

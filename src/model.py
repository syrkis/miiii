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

# constants
dataset = "primes"


def make_apply_fn(transformer_fn):  # x: seq_len
    @partial(vmap, in_axes=(None, 0))
    def apply_fn(params, x):  # set dropout_rate to 0.1 when training
        z = embed_fn(x, params["tok_emb"], params["pos_emb"])  # z: seq_len x emb_dim
        # z = lax.scan(transformer_fn, z, params["blocks"])  # z: seq_len x emb_dim
        for block in params["blocks"]:  # use fori_loop maybe
            z = transformer_fn(z, block)  # use different transformers
        logits = z @ params["lm_head"]  # logits: seq_len x vocab
        return logits[-1]

    return apply_fn


def layer_norm(x, eps=1e-6):  # i think this is wrong
    return x
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    return (x - mean) / (std + eps)


def dropout_fn(rng, x):
    rate = 0.1
    return random.bernoulli(rng, 1 - rate, x.shape) / (1 - rate)


def embed_fn(x, tok_emb_w, pos_emb_w):  # x: seq_len
    tok_emb = tok_emb_w[x]  # tok_emb: seq_len x emb_dim
    pos_emb = pos_emb_w[jnp.arange(x.shape[0])]  # pos_emb: seq_len x emb_dim
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def ffwd_fn(x, params):
    w1, b1, w2, b2 = params
    z = x @ w1 + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2
    return z


######################################
# Vaswani Transformer
######################################

vaswani_ffwd_fn = ffwd_fn


def vaswani_fn(z, block):
    z += vaswani_head_fn(layer_norm(z), block["head"])
    z += vaswani_ffwd_fn(layer_norm(z), block["ffwd"])
    return z


def vaswani_head_fn(x, params):
    query, key, value, projection = params

    mask = jnp.triu(jnp.full((x.shape[0], x.shape[0]), -jnp.inf), 1)
    q, k, v = x @ query, x @ key, x @ value  # q, k, v: seq_len x d_k
    z = q @ rearrange(k, "b t c -> b c t")  # z: seq_len x seq_len
    z /= jnp.sqrt(k.shape[-1])
    wei = jnp.where(dataset == "ficciones", z + mask, z)
    wei = jax.nn.softmax(wei, axis=-1)
    z = wei @ v  # z: head x seq_len x d_v
    z = rearrange(z, "h t d -> t (h d)")
    z = z @ projection
    return z


######################################
# Hosseini Transformer
######################################


######################################
# He transformer
######################################


# testing
if __name__ == "__main__":
    from utils import load_conf
    from param import init_fn
    from datum import data_fn, text_fn
    from numbs import base_n
    from oeis import oeis

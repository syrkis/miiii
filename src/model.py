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
    for block in params["blocks"]:  # use fori_loop maybe
        z = transformer_fn(z, block)  # use different transformers
    logits = z @ params["lm_head"]  # logits: seq_len x vocab
    return logits


def transformer_fn(z, block):
    z = head_fn(z, *block["head"])  # <- this the hardcore stuff
    # z += ffwd_fn(z, *block["ffwd"])
    return z


def head_fn(x, query, key, value):  # x: seq_len x emb_dim
    mask = jnp.triu(jnp.full((x.shape[0], x.shape[0]), -jnp.inf), 1)
    q = x @ query  # q: seq_len x d_k
    k = x @ key  # k: seq_len x d_k
    v = x @ value  # v: seq_len x d_v
    z = q @ rearrange(k, "b t c -> b c t")  # z: seq_len x seq_len
    z /= jnp.sqrt(k.shape[-1])
    wei = z + mask  # wei: seq_len x seq_len  # for decoder only
    wei = jax.nn.softmax(wei, axis=-1)
    z = wei @ v  # z: head x seq_len x d_v
    z = rearrange(z, "h t d -> t (h d)")
    return z


# CORRECT
def ffwd_fn(x, w1, b1, w2, b2):
    z = x @ w1 + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2
    z = jax.nn.relu(z)
    return z


# CORRECT
def embed_fn(x, tok_emb_w, pos_emb_w):  # x: seq_len
    tok_emb = tok_emb_w[x]  # tok_emb: seq_len x emb_dim
    pos_emb = pos_emb_w[jnp.arange(x.shape[0])]  # pos_emb: seq_len x emb_dim
    return tok_emb + pos_emb  # z: seq_len x emb_dim


if __name__ == "__main__":
    from utils import load_conf
    from param import init_fn
    from datum import data_fn, text_fn
    from numbs import base_n
    from oeis import oeis

    rng, key = random.split(random.PRNGKey(0))
    # ns = partial(base_n, n=2)
    # x, y = data_fn(oeis["A000040"], 2**10, ns)
    data, encode, decode, vocab = text_fn(key, 64, 4)  # block_size, batch_size
    config = dict(in_d=len(vocab), out_d=len(vocab), len=64, **load_conf())
    params = init_fn(key, config)
    pred = apply_fn(params, next(data)[0])
    print(pred)

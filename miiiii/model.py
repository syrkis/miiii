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
# mask = jnp.triu(jnp.full((x.shape[0], x.shape[0]), -jnp.inf), 1


def predict_fn(apply_fn, params, x):
    logits = apply_fn(params, random.PRNGKey(0), x, 0.0)
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


# optional rng
def make_apply_fn(transformer_fn):  # x: seq_len
    @partial(vmap, in_axes=(None, None, 0, None))
    def apply_fn(params, rng, x, dropout=0.0):  # set dropout_rate to 0.1 when training
        z = embed_fn(x, params["tok_emb"], params["pos_emb"])  # z: seq_len x emb_dim
        z, rng = dropout_fn(rng, z, dropout)
        for block in params["blocks"]:  # use fori_loop maybe
            z = transformer_fn(z, block)  # use different transformers
            z, rng = dropout_fn(rng, z, dropout)
        z = jnp.mean(z, axis=0)  # pool: emb_dim
        logits = z @ params["lm_head"]  # logits: seq_len x vocab
        return logits.squeeze()  # logits: vocab

    return apply_fn


def dropout_fn(rng, x, rate):
    rng, key = random.split(rng)
    return random.bernoulli(key, 1 - rate, x.shape) / (1 - rate) * x, rng


def embed_fn(x, tok_emb_w, pos_emb_w):  # x: seq_len
    tok_emb = tok_emb_w[x]  # tok_emb: seq_len x emb_dim
    pos_emb = pos_emb_w[jnp.arange(x.shape[0])]  # pos_emb: seq_len x emb_dim
    return tok_emb + pos_emb  # z: seq_len x emb_dim


def ffwd_fn(x, params):
    w1, w2, b1, b2 = params
    z = x @ w1 + b1  # z: seq_len x emb_dim
    z = jax.nn.relu(z)  # TODO: maybe switch activation
    z = z @ w2 + b2  # disable biases as per @nanda2023
    return z


# function for actually classifyinf (use sigmoid)
def classify_fn(logits):
    return (jax.nn.sigmoid(logits) > 0.5).astype(int)


""" def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta """


######################################
# Vaswani Transformer
######################################

vaswani_ffwd_fn = ffwd_fn


def vaswani_fn(z, block):
    z += vaswani_head_fn(z, block["head"])
    z += vaswani_ffwd_fn(z, block["ffwd"])
    return z


def vaswani_head_fn(x, params):
    query, key, value, projection = params
    q, k, v = x @ query, x @ key, x @ value  # q, k, v: seq_len x d_k
    z = q @ rearrange(k, "b t c -> b c t")  # z: seq_len x seq_len
    z /= jnp.sqrt(k.shape[-1])
    wei = jax.nn.softmax(z, axis=-1)
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
    from utils import get_conf
    from param import init_fn
    from datum import prime_fn
    from numbs import base_ns

    from oeis import oeis

    seq = oeis["A000040"]  # "A000040" is the sequence of prime numbers
    cfg = get_conf()
    alpha = (cfg["n"] / jnp.log(cfg["n"])) / cfg["n"]
    rng, key = random.split(random.PRNGKey(0))

    (x_train, y_train), _ = prime_fn(cfg["n"], cfg["base"], base_ns, key)

    params = init_fn(key, cfg, x_train, y_train)
    apply_fn = make_apply_fn(vaswani_fn)
    pred = apply_fn(params, key, x_train, 0.1)
    print(pred.shape, pred)

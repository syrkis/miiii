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
        z = jnp.mean(z, axis=0)  # pool: emb_dim
        logits = z @ params["lm_head"]  # logits: seq_len x vocab
        return logits.squeeze()  # logits: vocab

    return apply_fn


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


# function for actually classifyinf (use sigmoid)
def classify_fn(logits):
    return (jax.nn.sigmoid(logits) > 0.5).astype(int)


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
    from utils import get_conf
    from param import init_fn
    from datum import data_fn, text_fn
    from numbs import base_n
    from oeis import oeis

    seq = oeis["A000040"]  # "A000040" is the sequence of prime numbers
    data_conf, model_conf = get_conf()
    alpha = (data_conf["n"] / jnp.log(data_conf["n"])) / data_conf["n"]
    rng, key = random.split(random.PRNGKey(0))

    number_system = partial(base_n, data_conf["base"])
    train_data, valid_data = data_fn("primes", seq, data_conf["n"], number_system)

    params = init_fn(key, dict(**model_conf, len=train_data[0].shape[1]))
    apply_fn = make_apply_fn(vaswani_fn)
    pred = apply_fn(params, train_data[0])
    print(pred.shape, pred)

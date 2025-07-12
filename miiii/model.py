# %% model.py
#   miiii model for jax
# by: Noah Syrkis

# Imports
from miiii.types import Params
from miiii.tasks import Dataset
from jax import random, nn, vmap, debug
import jax.numpy as jnp
from jax import Array
from functools import partial


# Constants
initializer: nn.initializers.Initializer = nn.initializers.he_normal()


# Forward
# @partial(vmap, in_axes=(None, None, 0))
# def apply(key: Array, params: Params, x) -> Array:
#     x = jnp.take(params.tok_emb, x, axis=0) + jnp.take(params.pos_emb, jnp.arange(x.shape[0]), axis=0)
#     x = jnp.dot(nn.relu(jnp.dot(x, params.w_i)), params.w_o)  # z: seq_len x emb_dim
#     return jnp.dot(x[-1], params.out_emb)  # TODO: mask some things to zero?


# Init
def init_fn(rng, cfg, ds) -> Params:
    a, k = nn.initializers.glorot_normal(), random.split(rng, 5)
    s: tuple = ((cfg.p + 1, cfg.d), (3, cfg.d), (cfg.d, cfg.d * 4), (cfg.d * 4, cfg.d), (ds.t, cfg.d, cfg.p))
    return Params(tok=a(k[2], s[2]), pos=a(k[3], s[3]), w_i=a(k[0], s[0]), w_o=a(k[1], s[1]), out=a(k[4], s[4]))


# OTHER #############################################################
# def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
# mask = random.bernoulli(key, 1 - dropout, x.shape)
# return jnp.where(dropout == 0.0, x, mask * x / (1 - dropout))

# %% model.py
#   miii model for jax

# %% Imports
from miiii.types import Params
from jax import random, nn
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Array
from miiii.types import Dataset


# %% Functions
def apply(params: Params, dropout, rng: Array, x: Array) -> Tuple[Array, Array]:
    keys = random.split(rng, 2)
    x = params.tok.take(x, axis=0) + params.pos.take(jnp.arange(3), axis=0)
    x = dropout_fn(keys[0], x + allu(params, x), dropout)  # <--- gets the people going
    x = dropout_fn(keys[1], x + (z := nn.relu(x @ params.i)) @ params.o, dropout)
    return jnp.dot(x[0], params.out), z


def allu(params: Params, z: Array) -> Array:  # <-- all you need
    k, v, q = z @ params.k, z @ params.v, z @ params.q
    qk = jnp.einsum("bth, bsh -> bts", q, k) / jnp.sqrt(params.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    return (wei @ v @ params.p).sum(0)


def dropout_fn(key: Array, x: Array, dropout: float) -> Array:
    mask = random.bernoulli(key, 1 - dropout, x.shape)
    return jnp.where(dropout == 0.0, x, mask * x / (1 - dropout))


def init_fn(rng, cfg, ds: Dataset) -> Params:
    f, k = nn.initializers.glorot_normal(), random.split(rng, 9)
    att_shp = (cfg.h, cfg.d, cfg.d // cfg.h)
    emb: tuple = f(k[0], (cfg.p + 1, cfg.d)), f(k[1], (3, cfg.d)), f(k[2], (ds.primes.size, cfg.d, cfg.p))
    mlp: tuple = f(k[3], (cfg.d, cfg.d * 4)), f(k[4], (cfg.d * 4, cfg.d))
    att: tuple = f(k[5], att_shp), f(k[6], att_shp), f(k[7], att_shp), f(k[8], (cfg.h, cfg.d // cfg.h, cfg.d))
    return Params(tok=emb[0], pos=emb[1], out=emb[2], k=att[0], q=att[1], v=att[2], p=att[3], i=mlp[0], o=mlp[1])

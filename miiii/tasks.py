# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import Array, random
from oeis import oeis
from miiii.types import Dataset


def task_fn(key: Array, ctx) -> Dataset:
    primes = jnp.array(oeis["A000040"][1 : ctx.p])[jnp.array(oeis["A000040"][1 : ctx.p]) <= ctx.p]
    x = jnp.stack((jnp.arange(ctx.n) % ctx.p, jnp.arange(ctx.n) // ctx.p, jnp.array(ctx.p).repeat(ctx.n)), axis=-1)

    # y for miiii and nanda
    y_miiii = (x[:, :-1] * jnp.array((1, ctx.p))[None,]).sum(-1)[None,] % primes[:-1, None]
    y = jnp.concat((y_miiii, (x[:, :-1].sum(-1) % ctx.p)[None,])).T  # concat miii and nanda

    # idxs and masks TODO: expand mask do deal with addition nanda task dim
    idxs = random.permutation(key, jnp.arange(ctx.p**2))  # permute the indices
    fact_mask = jnp.tile(jnp.arange(primes.max()), primes.size).reshape((primes.size, -1)) < primes[:, None]  # factor
    task_mask = (1 / jnp.log(fact_mask.sum(-1))).at[-1].set(0)  # seems right

    # return dataset
    return Dataset(x=x, y=y, idxs=idxs, fact_mask=fact_mask, task_mask=task_mask / task_mask.sum(), primes=primes)

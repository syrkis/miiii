# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import Array, random
from oeis import oeis
from miiii.types import Dataset


def task_fn(key: Array, cfg) -> Dataset:
    # inclusive list of primes from 2 to cfg.p
    primes = jnp.int32(oeis["A000040"][1 : cfg.p])[jnp.array(oeis["A000040"][1 : cfg.p]) <= cfg.p]

    # shuffeling permutation
    idxs = random.permutation(key, jnp.arange(cfg.p**2))  # permute the indices

    # x vector in base p (x_0 + x_1 = _)
    x = jnp.vstack((idxs.sort() % cfg.p, idxs.sort() // cfg.p, jnp.array(cfg.p).repeat(cfg.p**2))).T

    # miiii task target (num samples times num primes less than cfg.p  (including p))
    y_miiii = (x[:, :-1] * jnp.array((1, cfg.p))).sum(-1, keepdims=True) % primes[:-1]

    # nanda task target vector
    y_nanda = x[:, :-1].sum(-1, keepdims=True) % cfg.p

    # joint y vector (could mask different sub tasks)
    y = jnp.concat((y_miiii, y_nanda), axis=-1)

    # mask away integers larger than the task in question
    mask = jnp.tile(jnp.arange(primes.max()), primes.size).reshape((primes.size, -1)) < primes[..., None]

    # weight submask relative to number of classes within it (correcting for expected loss)
    task = jnp.log(mask.sum(-1))  #  * jnp.ones(mask.shape[0])

    # final ds to return
    return Dataset(x=x, y=y, idxs=idxs, task=task, mask=mask, primes=primes, frac=cfg.frac)

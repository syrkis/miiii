# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import Array, random
from oeis import oeis
from miiii.types import Dataset, Split


def task_fn(key: Array, p) -> Dataset:
    # inclusive list of primes from 2 to p
    primes = jnp.int32(oeis["A000040"][1:p])[jnp.array(oeis["A000040"][1:p]) <= p]

    # shuffeling permutation
    idxs = random.permutation(key, jnp.arange(p**2))  # permute the indices

    # x vector in base p (x_0 + x_1 = _)
    x = jnp.int32(jnp.vstack((idxs.sort() % p, idxs.sort() // p, jnp.ones(p**2) * p)).T)

    # miiii task target (num samples times num primes less than p  (including p))
    y_miiii = (x[:, :-1] * jnp.array((1, p))).sum(-1, keepdims=True) % primes[:-1]

    # nanda task target vector
    y_nanda = x[:, :-1].sum(-1, keepdims=True) % p

    # joint y vector (could mask different sub tasks)
    y = jnp.concat((y_miiii, y_nanda), axis=-1)

    # mask away integers larger than the task in question
    classes = (jnp.tile(jnp.arange(p), (primes.size, 1)) < primes[:, None])[None, ...]

    # weight submask relative to number of classes within it (correcting for expected loss)
    weight = jnp.log(classes.sum(-1))  #  * jnp.ones(mask.shape[0])

    # how large fraction of ds to use for train
    limit = int(0.5 * idxs.size)

    # train and valid splits
    train, valid = Split(x=x[idxs][:limit], y=y[idxs][:limit]), Split(x=x[idxs][limit:], y=y[idxs][limit:])

    return Dataset(idxs=idxs, weight=weight, classes=classes, primes=primes, train=train, valid=valid)

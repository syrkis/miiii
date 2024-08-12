# data.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax.numpy as jnp
from jax import random
from jax import Array
from typing import Callable
from oeis import oeis


# %% Functions
def data_fn(n: int, base, ns_fn: Callable, key) -> mi.types.Dataset:
    primes = primes_fn(n)
    x_range = jnp.arange(2, n + 2)[:n]  # all numbers up to n
    x = ns_fn(mi.utils.digit_fn, x_range, base)  # number system representation of x

    # targets
    is_prime = jnp.zeros_like(x[:, 0]).at[primes - 2].set(1)[:, None]  # what numbers are prime
    primes_less_than_sqrt_n = primes[primes < jnp.sqrt(n)]   # prime numbers less than sqrt(n) for task
    is_multiple = x_range[:, None] % primes_less_than_sqrt_n == 0  # what numbers are multiples of other primes
    y = jnp.concatenate([is_prime, is_multiple], axis=-1)  # n x sqrt(n) + 1  # Concatenate tasks

    # shuffle data
    idxs = random.permutation(key, len(x))  # shuffle indices
    # x, y = x[idxs], y[idxs]  # shuffle data
    sep = len(x) // 2  # TODO: this a choise (split could be non 50/50
    tasks = ["prime"] + primes_less_than_sqrt_n.tolist()  # task labels

    # dataset
    dataset = mi.types.Dataset(
        train=mi.types.Datasplit(x=x[:sep], y=y[:sep]),
        valid=mi.types.Datasplit(x=x[sep:], y=y[sep:]),
        info=mi.types.Datainfo(apriori=y.mean(axis=0), tasks=tasks),
    )

    return dataset


def primes_fn(n: int) -> Array:
    limit = (n / jnp.log(n)).astype(jnp.int32)  # num primes less than n is n / ln(n)
    primes = jnp.array(oeis["A000040"][1 : limit * 2])  # get first limit primes
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    return primes


if __name__ == "__main__":
    print("data.py")

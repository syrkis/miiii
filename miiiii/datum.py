# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
import os
from jax import jit, random, vmap
import numpy as np
import requests
from tqdm import tqdm
from typing import List, Set, Tuple, Callable
from oeis import oeis
from functools import partial


# Constants
SPECIAL_TOKENS = {}  # {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2}


# classification related functions
def prime_fn(n: int, base, ns_fn: Callable, key) -> Tuple[jnp.array, jnp.array]:
    ns = partial(ns_fn, base)
    limit = (n / jnp.log(n)).astype(jnp.int32)  # num primes less than n is n / ln(n)
    primes = jnp.array(oeis["A000040"][1 : limit * 2])
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    x = ns(jnp.arange(2, n + 2)[:n])  # all numbers up to n

    # targets
    is_prime = jnp.zeros_like(x[:, 0]).at[primes - 2].set(1)[:, None]  # n x 1
    is_multi = (  # one_hot indicating of number is multiple of 2, 3, 4, ..., sqrt(n)
        jnp.arange(2, n + 2)[:n][:, None] % jnp.arange(2, jnp.sqrt(n) + 2) == 0
    ).astype(jnp.int32)  # n x sqrt(n)
    y = jnp.concatenate([is_prime, is_multi], axis=-1)  # n x sqrt(n) + 1
    print(is_prime.shape, is_multi.shape, y.shape)
    exit()

    # shuffle data
    idxs = random.permutation(key, len(x))
    x, y = x[idxs], y[idxs]

    # split data
    sep = len(x) // 2
    train_data = x[:sep], y[:sep]
    valid_data = x[sep:], y[sep:]
    return train_data, valid_data


# operator related functions
def operator_fn(operator: Callable, n: int) -> jnp.array:
    a, b = jnp.arange(n), n - jnp.arange(n)  # TODO: all combinations
    return jnp.stack([a, b, operator(a, b)], axis=-1)


addition_fn = lambda a, b: a + b
subtraction_fn = lambda a, b: a - b
division_fn = lambda a, b: a / b
multiplication_fn = lambda a, b: a * b
modulus_fn = lambda a, b: a % b
exponentiation_fn = lambda a, b: a**b


# testing
if __name__ == "__main__":
    from numbs import base_ns
    from utils import get_conf

    cfg = get_conf()
    rng = random.PRNGKey(0)
    data = prime_fn(cfg.n, cfg.base, base_ns, rng)

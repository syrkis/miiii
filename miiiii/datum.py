# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import miiiii
import math
import jax.numpy as jnp
from jax import Array
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
def data_fn(
    n: int, base, ns_fn: Callable, key
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    limit = (n / jnp.log(n)).astype(jnp.int32)  # num primes less than n is n / ln(n)
    primes = jnp.array(oeis["A000040"][1 : limit * 2])
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    x_range = jnp.arange(2, n + 2)[:n]  # all numbers up to n
    x = ns_fn(miiiii.utils.digit_fn, x_range, base)

    # targets
    is_prime = jnp.zeros_like(x[:, 0]).at[primes - 2].set(1)[:, None]  # n x 1
    primes_less_than_sqrt_n = primes[primes < jnp.sqrt(n)]
    is_multiple = x_range[:, None] % primes_less_than_sqrt_n == 0  # n x sqrt(n)
    y = jnp.concatenate([is_multiple, is_prime], axis=-1)  # n x sqrt(n) + 1

    # shuffle data
    idxs = random.permutation(key, len(x))
    x, y = x[idxs], y[idxs]

    # split data
    sep = len(x) // 2  # TODO: this is a choice
    train_data = x[:sep], y[:sep]
    valid_data = x[sep:], y[sep:]

    # TODO: consider making test data all numbers larger than n but represented by same number of digits
    tasks = primes_less_than_sqrt_n.tolist() + ["is_prime"]
    return train_data, valid_data, tasks


# operator related functions
def operator_fn(operator: Callable, n: int) -> Array:
    a, b = jnp.arange(n), n - jnp.arange(n)  # TODO: all combinations
    return jnp.stack([a, b, operator(a, b)], axis=-1)

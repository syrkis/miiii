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
def prime_fn(n: int, number_system: Callable, key) -> Tuple[jnp.array, jnp.array]:
    limit = (n / jnp.log(n)).astype(int)  # num primes less than n is n / ln(n)
    primes = jnp.array(oeis["A000040"][1 : limit * 2])
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    x = number_system(jnp.arange(2, n + 2)[:n])  # all numbers up to n
    y = jnp.zeros_like(x[:, 0]).at[primes - 2].set(1)
    idxs = random.permutation(key, len(x))
    x, y = x[idxs], y[idxs]
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
    pass

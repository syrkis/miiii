# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
from jax import jit, random, vmap
import numpy as np
from tqdm import tqdm
from typing import List, Set, Tuple, Callable
from oeis import oeis


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


# classification related functions
def classify_fn(rng: jnp.array, seq_id: str, n: int) -> Tuple[jnp.array, jnp.array]:
    seq = oeis[seq_id]
    seq_set = jnp.array(seq[seq.offset : n + seq.offset])
    rng, com_set = comp_fn(rng, seq_set, n)
    x = jnp.concatenate([seq_set, com_set])
    y = jnp.array([1] * len(seq_set) + [0] * len(com_set))
    idx = random.permutation(rng, len(x))
    return rng, x[idx], y[idx]


def comp_fn(rng: jnp.array, seq_set: Set[int], n: int) -> Tuple[jnp.array, jnp.array]:
    # generate a set of numbers that are not in seq_set
    com_set, seq_set = set(), set(seq_set.tolist())
    while len(com_set) < n:
        rng, key = random.split(rng)
        sample = random.randint(key, (n,), min(seq_set), max(seq_set) + 1)
        com_set |= set(sample.tolist())
        com_set -= seq_set
        com_set = set(list(com_set)[:n])
    return rng, jnp.array(list(com_set))


# continuation related functions
def continue_fn(rng: jnp.array, seq_id: str, n: int) -> jnp.array:
    seq = oeis[seq_id]
    seq_set = jnp.array(seq[seq.offset : n + seq.offset])
    return rng, seq_set

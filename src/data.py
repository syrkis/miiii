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


# conrad data function
def conrad_fn(rng, seq_len) -> None:
    # f_dir is in data dir of parent of current file
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    f_name = os.path.join(parent_dir, "data", "nostromo.txt")
    with open(f_name, "r") as f:
        text = f.read()
    c2i = {c: idx for idx, c in enumerate(sorted(list(set(text))))}
    i2c = {idx: c for c, idx in c2i.items()}
    toks = jnp.array([c2i[c] for c in text])[: len(text) // seq_len * seq_len]
    data = toks.reshape(-1, seq_len)
    idxs = random.permutation(rng, data.shape[0])
    return data[idxs][:128], c2i, i2c


# testing
if __name__ == "__main__":
    rng = random.PRNGKey(0)
    data, c2i, i2c = conrad_fn(rng, 32)

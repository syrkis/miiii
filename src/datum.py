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


# get data
def data_fn(dataset, *args):
    return text_fn(*args) if dataset == "ficciones" else prime_fn(*args)


# classification related functions
def prime_fn(seq, n: int, ns: Callable) -> Tuple[jnp.array, jnp.array]:
    limit = (n / jnp.log(n)).astype(int)  # num primes less than n is n / ln(n)
    primes = jnp.array(seq[1 : limit * 2])
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    x = jnp.arange(2, n + 2)[:n]  # all numbers up to n
    y = jnp.zeros_like(x).at[primes - 2].set(1)[:n]
    return ns(x), y  # ns is number system


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


# borges data function
def text_fn(rng, block_size=8, batch_size=64):
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    f_name = os.path.join(parent_dir, "data", "ficciones.txt")

    with open(f_name, "r", encoding="utf-8") as f:
        text = "".join([c for c in f.read() if c.isalnum() or c in " .,;:!?-()\n"])
    vocab = list(SPECIAL_TOKENS.keys()) + sorted(list(set(text)))

    stoi = {c: idx for idx, c in enumerate(vocab)}
    itos = {idx: c for c, idx in stoi.items()}

    encode = lambda x: jnp.array([stoi[c] for c in x])
    decode = lambda x: "".join([itos[idx] for idx in x])

    def aux(rng, toks):
        while True:
            rng, key = random.split(rng)
            idxs = random.randint(key, (batch_size,), 0, len(toks) - block_size)
            x = jnp.stack([toks[idx : idx + block_size] for idx in idxs])
            y = jnp.stack([toks[idx + 1 : idx + block_size + 1] for idx in idxs])
            yield x, y

    train_data = aux(rng, encode(text[: int(len(text) * 0.8)]))
    valid_data = aux(rng, encode(text[int(len(text) * 0.8) :]))

    return train_data, valid_data, encode, decode, vocab


# testing
if __name__ == "__main__":
    from numbs import base_n

    """     ns = partial(base_n, n=16)
    x, y = data_fn(oeis["A000040"], 2**4 - 2, ns)
    print(x, y, sep="\n") """
    data, encode, decode, vocab = text_fn(random.PRNGKey(0), 64, 4)
    print(len(vocab))

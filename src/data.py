# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
from jax import jit, random, vmap
from tqdm import tqdm

# functions
def prime_iterator_fn():
    D, q = {}, 2
    while True:
        if q not in D:  # q is a new prime.
            yield q     # Yield it and mark its first multiple that isn't already marked.
            D[q * q] = [q]
        else:           # q is not a prime. q is a composite number.
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]    # Remove this number, it's no longer needed.
        q += 1

def data_fn(n):
    prime_iter = prime_iterator_fn()
    primes = []
    for _ in range(n):
        p = next(prime_iter)
        primes.append(p)
    return jnp.array(primes)

# scrapped code generating [1, 3, 7, 9] context for each prime of partcular digitlength (too base 10 focused).
def d_data_fn(d):
    prime_iter = prime_iterator_fn()
    primes     = d_digit_primes(prime_iter, d - 1)
    context    = context_fn(primes)
    x          = context            # n x 4
    y          = jnp.equal(x, primes).astype(jnp.uint32)
    data       = jnp.concatenate([x, y], axis=1)
    rng        = random.PRNGKey(0)
    idxs       = random.permutation(rng, jnp.arange(len(data)))
    return data[idxs].astype(jnp.uint32)


def d_digit_primes(prime_iter, d):
    primes = []
    if d is not None:
        for _ in range(100_000):
            p = next(prime_iter)
            if p < 10**d:
                continue
            if p > 10**(d+1):
                break
            primes.append(p)
    return jnp.array(primes)[:, None]

@vmap
def context_fn(p):
    floor = 10 * jnp.floor(p / 10)
    context = [floor + 1, floor + 3, floor + 7, floor + 9]
    return jnp.array(context).squeeze()
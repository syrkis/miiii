# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
from jax import jit, random, vmap
from tqdm import tqdm

# functions
def prime_fn():
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

def primes_fn(n):
    prime_iter = prime_fn()
    primes = [next(prime_iter) for _ in range(n)]
    return jnp.array(primes)

def data_fn(n, d):
    primes = primes_fn(n)
    context = vmap(context_fn)(primes)
    return context

def context_fn(p):
    floor = 10 * jnp.floor(p / 10)
    context = [floor + 1, floor + 3, floor + 7, floor + 9]
    return jnp.array(context).squeeze()
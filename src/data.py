# data.py
#   miuii data functions
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

def prime_fn(digits=6):
    prime_iterator = prime_iterator_fn()
    primes = []
    for _ in range(100_000):
        p = next(prime_iterator)
        if p < 10**digits:
            continue
        if p > 10**(digits+1):
            break
        primes.append(p)
    return jnp.array(primes)[:, None]

@vmap
def context_fn(p):
    floor = 10 * jnp.floor(p / 10)
    context = [floor + 1, floor + 3, floor + 7, floor + 9]
    return jnp.array(context).squeeze()

def data_fn(digits=5):
    primes  = prime_fn(digits)
    context = context_fn(primes)
    x = context
    y = jnp.argmax(x == primes, axis=1).reshape(-1, 1)
    data = jnp.concatenate([x, y], axis=1)
    rng = random.PRNGKey(0)
    idxs = random.permutation(rng, jnp.arange(len(data)))
    return data[idxs].astype(jnp.uint32)
# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
from jax import jit, random, vmap
from tqdm import tqdm

# functions
def prime_fn():  # generates prime numbers
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
    
def primes_fn(d):  # returns a jnp.array of all d-digit primes
    prime_iter = prime_fn()
    primes     = []
    prime      = next(prime_iter)
    while prime < 10 ** d:
        if prime >= 10 ** (d - 1):
            primes.append(prime)
        prime = next(prime_iter)
    return jnp.array(primes).astype(jnp.int32)

def data_fn(d):  # returns a marix of all d digit numbers ending in 1,3,7, or 9, in a sorted (-1 x 4 x d)
    primes  = primes_fn(d)                                       # (n_primes,  )
    context = context_fn(d)                                      # (10 ** d , 4)
    y       = vmap(lambda row: target_fn(primes, row))(context)  # (10 ** d, 4)
    x       = vmap(lambda row: extract_digits(row, d))(context)  # (10 ** d, 4, 4)
    return x, y

def extract_digits(row, d):
    # each entry in row is a digit. it should be a vector of powers of 10
    powers_of_ten = 10 ** jnp.arange(d-1, -1, -1)
    digits        = row[:, None] // powers_of_ten[None, :] % 10
    return digits

def target_fn(primes, row):
    # return vector of 4 booleans, indicating if the row contains the prime numbers
    return jnp.any(row[None, :] == primes[:, None], axis=0).astype(jnp.int32)

def context_fn(d):
    context = jnp.arange(10 ** (d - 1), 10 ** d).reshape(-1, 10)
    context = context[:, [1, 3, 7, 9]]
    return context

def batch_fn(rng, data):
    x, y = data
    while True:
        rng, key = random.split(rng)
        idxs = random.permutation(key, x.shape[0])
        yield x[idxs], y[idxs]
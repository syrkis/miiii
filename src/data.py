# data.py
#   miii data functions
# by: Noah Syrkis

# Imports
import math
import jax.numpy as jnp
import numpy as np
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


# convert base ten int d to base n
def base_fn(digit, base=10):
    if digit == 0:
        return "0"
    digits = []
    while digit:
        remainder = digit % base
        if remainder > 9:
            # Map 10-15 to 'A'-'F' for base 16, and similarly for other bases
            digits.append(chr(55 + remainder))
        else:
            digits.append(str(remainder))
        digit //= base
    return ''.join(digits[::-1])

def data_fn(config):
    prime  = prime_fn()
    primes = jnp.array(list(set([next(prime) for _ in range(config['n_primes'])])))
    x      = jnp.arange(primes.max())
    y      = jnp.zeros_like(x).at[primes].add(1)
    return repr_fn(x, config), y
def repr_fn(x, config):
    if config['repr'] == 'positional':
        return position_fn(x, config)
    if config['repr'] == 'surreal':
        pass
    if config['repr'] == 'fibonacci':
        pass
    return x

def position_fn(x, config):
    def split_number_base_n(n, base, length): # Split an individual number into its digits for base-n
        return jnp.array([(n // (base ** i)) % base for i in range(length - 1, -1, -1)])
    max_length = jnp.max(jnp.log(x + 1) / jnp.log(config['base'])).astype(int) + 1
    split_vect = vmap(split_number_base_n, in_axes=(0, None, None), out_axes=0)
    return split_vect(x, config['base'], max_length)

def main():  # currently vocab = base (might want to generalise), toks=10)
    config = dict(n_primes=100, repr='positional', base=16)
    data   = data_fn(config)
    print(data)

if __name__ == '__main__':
    main()
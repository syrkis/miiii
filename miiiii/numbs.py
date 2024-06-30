# numbs.py
#   miiii number systems
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


# functions
def base_ns(digit_fn, base, x):
    digits = digit_fn(x.max(), base)
    numb = jnp.array([x // base**i % base for i in range(digits)][::-1]).T
    return numb


def fibo(x):
    pass


def prime(x):
    pass


if __name__ == "__main__":
    x = jnp.arange(33).T
    print(base_ns(x, 16))

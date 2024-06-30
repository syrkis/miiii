# numbs.py
#   miiii number systems
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


# functions
digit_fn = lambda n, base: jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(int)


def base_ns(base, x):
    digits = digit_fn(x.max(), base)
    numb = jnp.array([x // base**i % base for i in range(digits)][::-1]).T
    return numb


def fibo(x):
    pass


def prime(x):
    pass


if __name__ == "__main__":
    x = jnp.arange(33)
    print(base_ns(x, 16))

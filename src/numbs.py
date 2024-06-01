# numbs.py
#   miiii number systems
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


# functions
def base_n(x, n):  # TODO: fix for base > 10
    nits = jnp.ceil(jnp.log(x.max() + 1) / jnp.log(n)).astype(int)
    numb = jnp.array([x // n**i % n for i in range(nits)][::-1]).T
    return numb


def fibo(x):
    pass


def prime(x):
    pass


if __name__ == "__main__":
    x = jnp.arange(33)
    print(base_n(x, 16))

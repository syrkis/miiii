# numbs.py
#   miiii number systems
# by: Noah Syrkis

# imports
import jax.numpy as jnp


# functions
def base_ns(x, base):
    digits = digit_fn(x.max(), base)
    numb = jnp.array([x // base**i % base for i in range(digits)][::-1]).T
    return numb


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# def fibo(x):
# pass
#
#
# def prime(x):
# pass
#

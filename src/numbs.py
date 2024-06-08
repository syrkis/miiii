# numbs.py
#   miiii number systems
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


if __name__ == "__main__":
    from utils import digit_fn
else:
    from .utils import digit_fn


# functions
def base_n(base, x):
    digits = digit_fn(x.max(), base)
    numb = jnp.array([x // base**i % base for i in range(digits)][::-1]).T
    return numb


def fibo(x):
    pass


def prime(x):
    pass


if __name__ == "__main__":
    x = jnp.arange(33)
    print(base_n(x, 16))

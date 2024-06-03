# algos.py
#   miiii primality test algorithms
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import jit, vmap


# functions
def trial_division(n):
    return (vmap(lambda x: n % x == 0)(jnp.arange(2, int(jnp.sqrt(n)) + 1))).sum() == 0


if __name__ == "__main__":
    n = 2**31 - 1
    print(trial_division(n))

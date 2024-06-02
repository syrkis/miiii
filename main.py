# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
import src


# functions
def main():
    rng, key = random.split(random.PRNGKey(0))
    config = src.get_conf()
    print(config)


if __name__ == "__main__":
    main()

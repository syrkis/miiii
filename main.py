# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from oeis import oeis
import jax.numpy as jnp
from jax import random, jit
from typing import List, Set, Tuple

from src import args_fn, operator_fn, modulus_fn
# functions


def main():
    args = args_fn()
    operator = modulus_fn
    data = operator_fn(operator, 100000)
    print(data)


if __name__ == "__main__":
    main()

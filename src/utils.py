# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import jax.numpy as jnp


# functions
def args_fn():
    parser = argparse.ArgumentParser(description="c2sim")
    # specify which script in src to run
    parser.add_argument("--script", type=str, default="main", help="script to run")
    return parser.parse_args()


def conf_fn():
    return {
        "n_primes": 1000,
        "repr": "positional",
        "base": 10,
        "emb_dim": 128,
        "block_size": 64,
        "n_layers": 3,
        "n_heads": 8,
        "scale": 1 / jnp.sqrt(128),
    }

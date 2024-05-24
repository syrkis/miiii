# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml


# functions
def args_fn():
    parser = argparse.ArgumentParser(description="c2sim")
    parser.add_argument("--sequence", type=str, default="A000040")
    return parser.parse_args()

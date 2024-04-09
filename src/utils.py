# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml

# constants
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# functions
def args_fn():
    parser = argparse.ArgumentParser(description="c2sim")
    # specify which script in src to run
    parser.add_argument("--script", type=str, default="main", help="script to run")
    return parser.parse_args()


def conf_fn():
    with open(f"{ROOT}/conf.yaml", "r") as file:
        conf = yaml.safe_load(file)
    return conf

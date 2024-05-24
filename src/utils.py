# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml


def load_conf():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "conf.yaml"), "r") as file:
        conf = yaml.safe_load(file)
    return conf


# functions
def args_fn():
    parser = argparse.ArgumentParser(description="c2sim")
    parser.add_argument("--sequence", type=str, default="A000040")
    return parser.parse_args()

# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml


# functions
encode = lambda d, x: jnp.array([d[c] for c in x])
decode = lambda d, x: "".join([d[i] for i in x])


def load_conf(vocab: int):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "conf.yaml"), "r") as file:
        conf = yaml.safe_load(file)
    conf["vocab"] = vocab
    return conf


# functions
def args_fn():
    parser = argparse.ArgumentParser(description="c2sim")
    parser.add_argument("--sequence", type=str, default="A000040")
    return parser.parse_args()

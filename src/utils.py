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


def load_conf():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "conf.yaml"), "r") as file:
        conf = yaml.safe_load(file)
    return conf


def get_conf(**kwargs):
    conf = load_conf()
    for key, value in kwargs.items():
        conf[key] = value
    return conf

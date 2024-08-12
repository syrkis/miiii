# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml
import pickle
from chex import dataclass


# constants
red = "#da3527"


# functions
def encode(d, x):
    return jnp.array([d[c] for c in x])


def decode(d, x):
    return "".join([d[i] for i in x])


# prime to composite ratio
def alpha_fn(n):
    return 1 - ((n / jnp.log(n)) / n)


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


@dataclass
class DataConf:
    base: int = 2
    n: int = 2**12
    emb: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 100
    gamma: int = 2
    l2: float = 1e-4  # lambda
    lr: float = 1e-3
    dropout: float = 0.1
    block: str = "vaswani"


def load_conf():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "config.yaml"), "r") as file:
        conf = yaml.safe_load(file)
    conf = DataConf(**conf)
    return conf


def get_conf(**kwargs):
    conf = load_conf()
    return conf


def get_args():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--n", type=int, default=2**14)
    return parser.parse_args()


def save_params(params, path):
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_params(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import miiiii.kinds as kinds
import argparse
import os
import jax.numpy as jnp
import yaml
import pickle


# constants
red = "#da3527"
blue = "#002fa7"


# functions
def load_conf():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "config.yaml"), "r") as file:
        conf = yaml.safe_load(file)
    conf = kinds.Conf(**conf)
    return conf


def save_params(params, fname):
    path = os.path.join("data", fname)
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_params(fname):
    path = os.path.join("data", fname)
    with open(path, "rb") as file:
        return pickle.load(file)

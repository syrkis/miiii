# utils.py
#   miiii utils
# by: Noah Syrkis

# imports
import argparse
import os
import jax.numpy as jnp
import yaml
import pickle


# functions
encode = lambda d, x: jnp.array([d[c] for c in x])
decode = lambda d, x: "".join([d[i] for i in x])

# prime to composite ratio
alpha_fn = lambda n: 1 - ((n / jnp.log(n)) / n)


def load_conf():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "config.yaml"), "r") as file:
        conf = yaml.safe_load(file)
        conf["in_d"] = conf["base"]
        conf["out_d"] = 1
    return conf


def get_conf(**kwargs):
    conf = load_conf()
    for key, value in kwargs.items():
        conf[key] = value
    return conf


def get_args():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--n", type=int, default=2**14)
    return parser.parse_args()


def save_model(params, path):
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    data_conf, model_conf = load_conf()
    print(data_conf)

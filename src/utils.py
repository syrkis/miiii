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
    with open(os.path.join(parent_dir, "conf/data.yaml"), "r") as file:
        data_conf = yaml.safe_load(file)
    with open(os.path.join(parent_dir, "conf/model.yaml"), "r") as file:
        model_conf = yaml.safe_load(file)
        model_conf["in_d"] = data_conf["base"]
        model_conf["out_d"] = 2
    return data_conf, model_conf


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


if __name__ == "__main__":
    data_conf, model_conf = load_conf()
    print(data_conf)

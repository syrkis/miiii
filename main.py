# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi

import jax.numpy as jnp
import yaml
from jax import random, tree


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    rng = random.PRNGKey(0)
    cfg = mi.utils.create_cfg(**config)
    ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
    state, metrics = mi.train.train(rng, cfg, ds, task)
    print(tree.map(jnp.shape, metrics))


if __name__ == "__main__":
    main()

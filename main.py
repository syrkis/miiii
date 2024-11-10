# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiii as mi
import sys
from jax import random


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    args = mi.utils.parse_args()
    cfg = mi.utils.create_cfg(args)
    rng, *keys = random.split(random.PRNGKey(0), 3)
    ds = mi.tasks.task_fn(keys[0], cfg)
    state, (metrics, acts) = mi.train.train(keys[1], cfg, ds)

    mi.utils.log_fn(cfg, ds, state, metrics, acts)

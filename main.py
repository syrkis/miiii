# %% main.py
#   miiii notebook
# by: Noah Syrkis

# %% Imports

# import esch
from jax import random

import miiii as mi

# %% Training
keys = random.split(random.PRNGKey(0), 2)
for project, task in [("nanda", "multi"), ("miiii", "multi"), ("miiii", "binary")]:
    kwargs = dict(project=project, task=task)
    cfg = mi.utils.create_cfg(**kwargs)
    ds = mi.tasks.task_fn(keys[0], cfg)  # create dataset
    state, (metrics, acts) = mi.train.train(keys[1], cfg, ds)  # train
    # mi.utils.log_fn(cfg, ds, state, metrics, acts)  # log

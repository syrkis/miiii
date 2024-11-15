# %% main.py
#   miiii notebook
# by: Noah Syrkis

# %% Imports
import miiii as mi

# import esch
from jax import random
from functools import partial
from itertools import product


# %% Training
def train_task_fn(rng, cfg, task_type, task_span):
    ds = mi.tasks.task_fn(rng, cfg, task_type, task_span)  # create dataset
    state, (metrics, acts) = mi.train.train(rng, cfg, ds)
    return state, (metrics, acts)


rng = random.PRNGKey(0)
tasks = list(product(["divisible", "remainder"], ["atomic", "batch"]))
cfg = mi.utils.create_cfg()
train_task = partial(train_task_fn, rng, cfg)
mi.utils.log_fn([train_task(*task) for task in tasks], cfg, tasks)

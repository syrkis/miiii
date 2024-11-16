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
    ds, task = mi.tasks.task_fn(rng, cfg, task_type, task_span)  # create dataset
    state, (metrics, acts) = mi.train.train(rng, cfg, ds, task)
    return state, (metrics, acts), task


rng = random.PRNGKey(0)
# tasks = list(product(["divisible", "remainder"], ["atomic", "batch"]))
tasks = [("remainder", "batch"), ("remainder", "atomic")]
cfg = mi.utils.create_cfg()
train_task = partial(train_task_fn, rng, cfg)
runs = list(map(lambda args: train_task(*args), tasks))
args = mi.utils.parse_args()
mi.utils.log_fn(runs, cfg)

# %% main.py
#   miiii notebook
# by: Noah Syrkis

# %% Imports

# import esch
from jax import random
from functools import partial
from itertools import product


import miiii as mi

# %% Training
# keys = random.split(random.PRNGKey(0), 2)
# cfgs = []
# dss = []
# statess, metricss, actss = [], [], []
# for project, task in [("nanda", "multi"), ("miiii", "multi"), ("miiii", "binary")]:
#     kwargs = dict(project=project, task=task)

#     cfg = mi.utils.create_cfg(**kwargs)
#     cfgs.append(cfg)

#     ds = mi.tasks.task_fn(keys[0], cfg)  # create dataset
#     dss.append(ds)

#     state, (metrics, acts) = mi.train.train(keys[1], cfg, ds)  # train
#     statess.append(state)
#     metricss.append(metrics)
#     actss.append(acts)


# for cfg, ds, state, metrics, acts in zip(cfgs, dss, statess, metricss, actss):
# mi.utils.log_fn(cfg, ds, state, metrics, acts)  # log


def train_task_fn(rng, cfg, task_type, task_span):
    ds = mi.tasks.task_fn(rng, cfg, task_type, task_span)  # create dataset
    state, (metrics, acts) = mi.train.train(rng, cfg, ds)
    return state, (metrics, acts)


rng = random.PRNGKey(0)
tasks = list(product(["divisible", "remainder"], ["atomic", "batch"]))
cfg = mi.utils.create_cfg()
train_task = partial(train_task_fn, rng, cfg)
mi.utils.log_fn([train_task(*task) for task in tasks], cfg, tasks)

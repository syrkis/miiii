# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi
from jax import random
import numpy as np
from einops import rearrange
import esch

# %% Globals
arg = mi.utils.arg_fn()
cfg = mi.utils.cfg_fn()


# study = mi.optim.run_study(arg)
def grid(arr):
    if arr.ndim == 2:
        arr = arr[:, :, None]
    dwg = esch.init(*arr.shape)
    esch.grid_fn(arr / arr.max(), dwg, dwg, shape="square")
    return dwg


# %% Setup
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg, arg)
tmp = np.array(rearrange(ds.y, "(a b) c -> c b a", a=cfg.p))[2::3]
dwg = esch.init(cfg.p, cfg.p, tmp.shape[0])
idx = 0
for idx in range(tmp.shape[0]):
    group = dwg.g()
    group.translate((tmp[idx].shape[1] + 1) * idx, 0)
    esch.grid_fn(tmp[idx] / tmp[idx].max(), dwg, group, shape="square")
    dwg.add(group)
esch.save(dwg, "/Users/nobr/desk/s3/miiii/out.svg")

# %%
# state, (scope, metrics, loss) = mi.train.train_fn(rng, cfg, arg, ds)
# scope = scope.transpose(3, 1, 2, 0)[:4]
# esch.grid(scope, path="miiii.svg")
# print(scope.shape)
# mi.utils.log_fn(cfg, arg, ds, state, metrics)
# tmp = rearrange(scope[0][:, 0, 0, :4], "(a b) n -> n a b 1", a=cfg.p)
# arr = np.array(tmp.squeeze()[0])
# dwg = esch.init(*arr.shape)
# esch.grid_fn(arr / arr.max(), dwg, dwg, shape="square")
# esch.save(dwg, "miiii.svg")

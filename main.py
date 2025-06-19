# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
import miiii as mi
from jax import random
import numpy as np
from einops import rearrange
import esch

# %% Constants
folder = "/Users/nobr/desk/s3/miiii"

# %% Globals
arg = mi.utils.arg_fn()
cfg = mi.utils.cfg_fn()

# %% Dataset
rng, key = random.split(random.PRNGKey(0))
ds = mi.tasks.task_fn(key, cfg, arg)

# %% Init
state, opt = mi.train.init_state(rng, cfg, arg, ds)
logits, z = mi.model.apply_fn(cfg, ds, 0.0)(rng, state.params, ds.x)

print(logits.shape, z.shape)
# state, (scope, metrics, loss) = mi.train.train_fn(rng, cfg, arg, ds)

exit()

# tmp = np.array(rearrange(ds.y, "(n p) t -> t n p", n=cfg.p, p=cfg.p))[-1][None, ...] / cfg.p
# e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=tmp.shape[0])
# print(tmp.shape)
# esch.grid_fn(tmp, e, shape="square")
# esch.save(e.dwg, f"{folder}/{cfg.p}_nanda_y.svg")

# %% Setup
# tmp = np.array(rearrange(ds.y, "(a b) c -> c b a", a=cfg.p))[2::3]
# dwg = esch.init(cfg.p, cfg.p, tmp.shape[0])
# idx = 0
# for idx in range(tmp.shape[0]):
# group = dwg.g()
# group.translate((tmp[idx].shape[1] + 1) * idx, 0)
# esch.grid_fn(tmp[idx] / tmp[idx].max(), dwg, group, shape="square")
# dwg.add(group)
# esch.save(dwg, "/Users/nobr/desk/s3/miiii/out.svg")

# %%
# scope = scope.transpose(3, 1, 2, 0)[:4]
# esch.grid(scope, path="miiii.svg")
# print(scope.shape)
# mi.utils.log_fn(cfg, arg, ds, state, metrics)
# tmp = rearrange(scope[0][:, 0, 0, :4], "(a b) n -> n a b 1", a=cfg.p)
# arr = np.array(tmp.squeeze()[0])
# dwg = esch.init(*arr.shape)
# esch.grid_fn(arr / arr.max(), dwg, dwg, shape="square")
# esch.save(dwg, "miiii.svg")

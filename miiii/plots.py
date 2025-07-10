# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import miiii as mi
import esch
from einops import rearrange
import numpy as np


folder = "/Users/nobr/desk/s3/miiii"


# %% plot y
def plot_y(cfg, ds):
    def anim():
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> 1 n p t", n=cfg.p, p=cfg.p), dtype=float))
        e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=1)
        esch.grid_fn(e, arr / arr.max((1, 2, 3), keepdims=True) * 0.5, shape="square", fps=1)
        esch.save(e.dwg, f"{folder}/{cfg.p}.svg")

    def mult():
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> 1 n p t", n=cfg.p, p=cfg.p), dtype=float))
        for idx, task in enumerate(arr):
            e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=1)
            esch.grid_fn(e, task[None, ...] / task.max((0, 1), keepdims=True) * 0.5, shape="square", fps=1)
            esch.save(e.dwg, f"{folder}/{cfg.p}_mod_{ds.primes[idx].item()}.svg")

    def slice():
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> n t p", n=cfg.p, p=cfg.p), dtype=float))
        for idx, batch in enumerate(arr):
            e = esch.Drawing(h=cfg.p - 1, w=ds.primes.size - 1, row=1, col=1)
            esch.grid_fn(e, batch[None, ...] / batch.max((0, 1), keepdims=True) * 0.5, shape="square", fps=1)
            esch.save(e.dwg, f"{folder}/{cfg.p}_batch_{idx}.svg")

    return [anim(), mult(), slice()]


# %% plot x
def plot_x(cfg, ds: mi.types.Dataset):
    arr = np.sqrt(np.array(rearrange(ds.x[:, :2], "(n p) d -> n p d", n=cfg.p, p=cfg.p)))
    e = esch.Drawing(w=cfg.p - 1, h=2 - 1, col=cfg.p, row=1)
    esch.grid_fn(e, arr / arr.max() * 0.9, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_x.svg")

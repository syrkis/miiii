# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import esch
from einops import rearrange
import numpy as np
import mlxp


folder = "/Users/nobr/desk/s3/miiii"


# %% plot y
def plot_y(**kwargs):
    cfg, ds = kwargs["cfg"], kwargs["ds"]

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
def plot_x(**kwargs):
    cfg, ds = kwargs["cfg"], kwargs["ds"]
    arr = np.sqrt(np.array(rearrange(ds.x[:, :2], "(n p) d -> n p d", n=cfg.p, p=cfg.p)))
    e = esch.Drawing(w=cfg.p - 1, h=2 - 1, col=cfg.p, row=1)
    esch.grid_fn(e, arr / arr.max() * 0.9, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_x.svg")


def plot_params(**kwargs):  # cfg, ds, params: mi.types.Params):
    cfg, params, ds = kwargs["cfg"], kwargs["params"], kwargs["ds"]
    arr = np.array(rearrange(np.abs(params.unbeds), "a b c -> b a c") / params.unbeds.max(), dtype=float) ** 0.5
    e = esch.Drawing(w=ds.primes.size - 1, h=cfg.p - 1, row=2, col=3, debug=False, pad=2)
    esch.grid_fn(e, arr, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_unbeds_x.svg")


def plot_log(**kwargs):
    parent_log_dir = "./logs/"
    reader = mlxp.Reader(parent_log_dir, refresh=True)
    query: str = "info.status == 'COMPLETE'"
    results = reader.filter(query_string=query, result_format="pandas")
    print(results["train.loss"])

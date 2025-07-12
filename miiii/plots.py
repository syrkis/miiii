# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import esch
from einops import rearrange
import numpy as np
import jax.numpy as jnp
from jax import tree
import mlxp
import seaborn as sns
import matplotlib.pyplot as plt


folder = "/Users/nobr/desk/s3/miiii"


# %% plot y
def plot_y(**kwargs) -> None:
    cfg, ds = kwargs["cfg"], kwargs["ds"]

    def anim() -> None:
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> 1 n p t", n=cfg.p, p=cfg.p), dtype=float))
        e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=1)
        esch.grid_fn(e, arr / arr.max((1, 2, 3), keepdims=True) * 0.5, shape="square", fps=1)
        esch.save(e.dwg, f"{folder}/{cfg.p}.svg")

    def mult() -> None:
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> 1 n p t", n=cfg.p, p=cfg.p), dtype=float))
        for idx, task in enumerate(arr):
            e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=1)
            esch.grid_fn(e, task[None, ...] / task.max((0, 1), keepdims=True) * 0.5, shape="square", fps=1)
            esch.save(e.dwg, f"{folder}/{cfg.p}_mod_{ds.primes[idx].item()}.svg")

    def slice() -> None:
        arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> n t p", n=cfg.p, p=cfg.p), dtype=float))
        for idx, batch in enumerate(arr):
            e = esch.Drawing(h=cfg.p - 1, w=ds.primes.size - 1, row=1, col=1)
            esch.grid_fn(e, batch[None, ...] / batch.max((0, 1), keepdims=True) * 0.5, shape="square", fps=1)
            esch.save(e.dwg, f"{folder}/{cfg.p}_batch_{idx}.svg")

    [anim(), mult(), slice()]


# %% plot x
def plot_x(**kwargs) -> None:
    cfg, ds = kwargs["cfg"], kwargs["ds"]
    arr = np.sqrt(np.array(rearrange(ds.x[:, :2], "(n p) d -> n p d", n=cfg.p, p=cfg.p)))
    e = esch.Drawing(w=cfg.p - 1, h=2 - 1, col=cfg.p, row=1)
    esch.grid_fn(e, arr / arr.max() * 0.9, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_x.svg")


def plot_params(**kwargs) -> None:  # cfg, ds, params: mi.types.Params):
    cfg, params, ds = kwargs["cfg"], kwargs["params"], kwargs["ds"]
    arr = np.array(rearrange(np.abs(params.out), "a b c -> b a c") / params.out.max(), dtype=float) ** 0.5
    e = esch.Drawing(w=ds.primes.size - 1, h=cfg.p - 1, row=2, col=3, debug=False, pad=2)
    esch.grid_fn(e, arr, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_unbeds_x.svg")


def plot_log(**kwargs) -> None:
    parent_log_dir = "./logs/"
    reader = mlxp.Reader(parent_log_dir, refresh=True)
    query: str = "info.status == 'COMPLETE'"
    # results = reader.filter(query_string=query, result_format="pandas")
    # tmp = np.array(results["scope.train_cce"].tolist())
    # sns.heatmap(tmp[-1].T)
    # plt.show()
    # tmp = jnp.array(results["train.loss"].to_list())
    # print(tmp.shape)
    # sns.heatmap(tmp)
    # plt.show()


def plot_metrics(**kwargs) -> None:
    cfg, params, ds, scope = kwargs["cfg"], kwargs["params"], kwargs["ds"], kwargs["scope"]
    acc = np.array(jnp.stack((scope.train_acc, scope.valid_acc)), dtype=float).transpose(0, 2, 1)
    cce = np.array(jnp.stack((scope.train_cce, scope.valid_cce)), dtype=float).transpose(0, 2, 1)

    data = np.array(np.concat((cce / cce.max(), acc / acc.max())), dtype=float)

    e = esch.Drawing(w=data.shape[1] - 1, h=data.shape[2] - 1, row=data.shape[0], col=1, pad=4)
    esch.grid_fn(e, np.sqrt(data), shape="square")
    esch.save(e.dwg, f"{folder}/{dict(**cfg)}_train.svg")


fns = [plot_params, plot_x, plot_y, plot_metrics]

# %% plot.py
#    miii plots


# %% imports
import esch
from einops import rearrange
import numpy as np
import jax.numpy as jnp

# from jax import tree
import mlxp
# import seaborn as sns
# import matplotlib.pyplot as plt


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

    # def slice() -> None:
    #     arr = np.sqrt(np.array(rearrange(ds.y, "(n p) t -> n t p", n=cfg.p, p=cfg.p), dtype=float))
    #     for idx, batch in enumerate(arr):
    #         e = esch.Drawing(h=cfg.p - 1, w=ds.primes.size - 1, row=1, col=1)
    #         esch.grid_fn(e, batch[None, ...] / batch.max((0, 1), keepdims=True) * 0.5, shape="square", fps=1)
    #         esch.save(e.dwg, f"{folder}/{cfg.p}_batch_{idx}.svg")

    [anim(), mult()]


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


def plot_metrics(**kwargs) -> None:
    cfg, params, ds, scope = kwargs["cfg"], kwargs["params"], kwargs["ds"], kwargs["scope"]

    def metrics() -> None:
        acc = np.array(jnp.stack((scope.train_acc, scope.valid_acc)), dtype=float).transpose(0, 2, 1)
        cce = np.array(jnp.stack((scope.train_cce, scope.valid_cce)), dtype=float).transpose(0, 2, 1)
        data = np.array(np.concat((cce / cce.max(), acc / acc.max())), dtype=float)
        e = esch.Drawing(w=data.shape[1] - 1, h=data.shape[2] - 1, row=data.shape[0], col=1, pad=4)
        esch.grid_fn(e, np.sqrt(data), shape="square")
        esch.save(e.dwg, f"{folder}/{cfg_to_str(cfg)}_train.svg")

    def period() -> None:
        neu = rearrange(np.array(scope.neu[..., -1]).real, "a b c -> 1 b c a")[:, 1:, 1:, ...].astype(float)
        fft = rearrange(np.array(scope.fft[..., -1]).real, "a b c -> 1 b c a")[:, 1:, 1:, ...].astype(float)
        data = np.concat((neu, fft)) + 0.0001
        e = esch.Drawing(w=data.shape[1] - 1, h=data.shape[2] - 1, row=1, col=data.shape[0], pad=4)
        esch.grid_fn(e, data / data.max((1, 2, 3), keepdims=True), shape="square", fps=1)
        esch.save(e.dwg, f"{folder}/{cfg_to_str(cfg)}_fft.svg")

    def all_periods() -> None:
        # neu = rearrange(np.abs(np.array(scope.neu[-1], dtype=float)), "a b c -> b c a")[:, 1:, 1:, ...]
        neu = np.array(rearrange(scope.neu, "t a b c -> c a b t"), dtype=float)[:, :16, :16]
        e = esch.Drawing(w=neu.shape[1] - 1, h=neu.shape[2] - 1, row=1, col=3, pad=4)
        esch.grid_fn(e, neu / neu.max(0, keepdims=True), shape="square", fps=1)
        esch.save(e.dwg, f"{folder}/{cfg_to_str(cfg)}_all_neu.svg")

    def money_shot() -> None:
        data = (scope.fft > (scope.fft.mean((1, 2), keepdims=True) + 2 * scope.fft.std((1, 2), keepdims=True))).sum(
            (-2, -1)
        )
        data = rearrange(np.array(data, dtype=float), "b c -> 1 c b")
        e = esch.Drawing(w=data.shape[1] - 1, h=data.shape[2] - 1, row=1, col=1, pad=4)
        esch.grid_fn(e, data / data.max(1, keepdims=True), shape="square")
        esch.save(e.dwg, f"{folder}/money_shot_{cfg_to_str(cfg)}.svg")

    [metrics(), period(), all_periods(), money_shot()]


def cfg_to_str(cfg):
    return "_".join(f"{k}-{v}" for k, v in cfg.items())


fns = [plot_params, plot_x, plot_y, plot_metrics]

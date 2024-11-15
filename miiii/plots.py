# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import datetime
import os
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

import miiii as mi

# %% Constants and configurations
fg = "black"
bg = "white"
plt.rcParams["font.family"] = "Monospace"
# set math text to new computer modern
plt.rcParams["mathtext.fontset"] = "cm"


# %% functions
def plot_run(metrics, ds: mi.tasks.Dataset, cfg: mi.utils.Conf, activations=None):
    # make run folder in figs/runs folder
    metrics = mi.utils.metrics_to_dict(metrics)
    # print(metrics["train"]["loss"].shape)
    # exit()
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"paper/figs/runs/{time_stamp}"  # _{name_run(cfg)}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/", exist_ok=True)

    # curve plots of all metrics
    for split in ["train", "valid"]:
        for metric in metrics[split]:
            curve_plot(metrics[split][metric], metric, path, split)
            plt.close()


########################################################################################
# %% Hinton plots


########################################################################################
# %% Polar plots
def polar_plot(ps: Sequence[Sequence] | Sequence | np.ndarray, f_name: Sequence[str] | str | None = None, ax=None):
    # init and config
    ax_was_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10), dpi=100)
    # limit should be from 0 to max of all ps
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # plot
    ps = ps if isinstance(ps[0], (Sequence, np.ndarray)) else [ps]  # type: ignore
    ax.plot([0, 0], "black", linewidth=1)
    for p in ps:
        ax.plot(p, p, "o", markersize=2, color="black")
    plt.tight_layout()
    plt.savefig(f"paper/figs/{f_name}.svg") if f_name else plt.show()
    if ax_was_none:
        plt.close()


def small_multiples(fnames, seqs, f_name, n_rows=2, n_cols=2):
    assert (
        len(fnames) == len(seqs) and len(fnames) >= n_rows * n_cols
    ), "fnames and seqs must be the same length and n_rows * n_cols"
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(n_cols * 5, n_rows * 5), dpi=100)
    for ax, fname, seqs in zip(axes.flat, fnames, seqs):  # type: ignore
        polar_plot(seqs, fname, ax=ax)
    # tight
    plt.tight_layout()
    plt.savefig(f"paper/figs/{f_name}.svg") if f_name else plt.show()  # test


########################################################################################
# %% Curve plots
def curve_plot(data, metric, path, split):
    fig, ax = init_curve_plot()
    new_len = len(data[0])
    for i, curve in enumerate(data):
        # smooth curve
        if i == len(data) - 1:
            curve = np.convolve(curve, np.ones(10) / 10, mode="valid")
            new_len = len(curve)
            ax.plot(curve, c=fg, lw=2, ls="--" if i > 0 else "-")
        else:
            continue
    # ax.set_xscale("log")  # make x-axis log
    # smoothing x axis
    ax.set_xlim(10, new_len)
    # add info key, val pairs to right side of plot (centered verticall on right axis) (relative to len(info))
    # ax.legend(info["legend"], frameon=False, labelcolor=fg)
    # make fname contain conf
    plt.tight_layout()
    plt.savefig(f"{path}/{split}_{metric}_curve.svg")


def init_curve_plot():
    # title is block with large first letter
    # title = conf["block"][0].upper() + conf["block"][1:] + " " + info["title"]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg)
    # make fg white
    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg)
    ax.set_facecolor(bg)
    ax.grid(False)
    # ax.set_title(title, color=fg)
    # ax.set_xlabel(info["xlabel"], color=fg)
    # ax.set_ylabel(info["ylabel"], color=fg)
    # font color of legend should also be fg
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    [spine.set_edgecolor(fg) for spine in ax.spines.values()]
    # ax y range from 0 to what it is
    return fig, ax


def horizontal_mean_pooling(x: Array, width: int = 3) -> Array:
    """Rolling mean array. Shrink to be rows x rows * width."""
    x = x[:, : (x.shape[1] // (x.shape[0] * width)) * (x.shape[0] * width)]
    i = jnp.eye(x.shape[0] * width).repeat(x.shape[1] // (x.shape[0] * width), axis=-1)
    z = (x[:, None, :] * i[None, :, :]).sum(axis=-1)
    return z / (x.shape[1] // (x.shape[0] * width))

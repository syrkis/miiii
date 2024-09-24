# %% plot.py
#    miii plots
# by: Noah Syrkis

"""
plot types:
    1. hinton plot for model weights
    2. syrkis plot for training progress of metrics
    3. polar plot for number sequences
    4. curve plot for training progress of loss
"""

# %% imports
import miiiii as mi
from jax import Array
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Sequence
import jax.numpy as jnp
from typing import Dict
import datetime

# %% Constants and configurations
fg = "black"
bg = "white"
plt.rcParams["font.family"] = "Monospace"
# set math text to new computer modern
plt.rcParams["mathtext.fontset"] = "cm"


# %% functions
def plot_run(metrics, ds: mi.kinds.Dataset, cfg: mi.kinds.Conf, activations=None):
    # make run folder in figs/runs folder
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"paper/figs/runs/{time_stamp}"  # _{name_run(cfg)}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/", exist_ok=True)

    # hinton plots of all metrics
    for split in ["train", "valid"]:  # if metrics is dictionary
        for metric in metrics[split]:  # if metrics is dictionary
            max_val = 1 if "f1" in metric else metrics["train"]["loss"].max().item()
            hinton_metric(metrics[split][metric], metric, ds, path, split, max_val, cfg)
            plt.close()
    # curve plots of all metrics
    for split in ["train", "valid"]:
        for metric in metrics[split]:
            curve_plot(metrics[split][metric], metric, path, split)
            plt.close()
    if activations is not None:
        for i, act in enumerate(activations):
            hinton_activations(act, path)
            plt.close()


def name_run(cfg: mi.kinds.Conf):
    datum_name = f"base_{cfg.base}_n_{cfg.n}"
    model_name = f"emb_{cfg.latent_dim}_heads_{cfg.heads}_depth_{cfg.depth}"
    train_name = f"lr_{cfg.lr}_epochs_{cfg.epochs}_l2_{cfg.l2}_dropout_{cfg.dropout}"
    name = f"{datum_name}_{model_name}_{train_name}"
    return name


def title_fn(cfg: mi.kinds.Conf):
    title = f"base : {cfg.base} | emb : {cfg.latent_dim} | heads : {cfg.heads} | depth : {cfg.depth} | lr : {cfg.lr} | l2 : {cfg.l2} | dropout : {cfg.dropout}"
    # replce " | " with "    |    "
    title = title.replace(" | ", "   |   ")
    return title


########################################################################################
# %% Hinton plots
def hinton_weight(weight: Array, path: str):
    assert len(weight.shape) >= 3
    fig, ax = plt.subplots(ncols=weight.shape[0], figsize=(12, 12))
    for i in range(weight.shape[0]):
        hinton_fn(weight[i], ax[i])
    plt.tight_layout()
    plt.savefig(f"{path}/svg/weights.svg")


def hinton_activations(acts: Array, path: str):
    assert len(acts.shape) == 3
    fig, axes = plt.subplots(nrows=acts.shape[0], figsize=(12, 12))
    for i in range(acts.shape[0]):
        hinton_fn(acts[i], axes[i])
    plt.tight_layout()
    plt.savefig(f"{path}/activations.svg")


def hinton_metric(
    data: Array, metric: str, ds: mi.kinds.Dataset, path: str, split: str, max_val: float, cfg: mi.kinds.Conf
):
    fig, ax = plt.subplots(figsize=(12, 5))
    pool_data = horizontal_mean_pooling(data)
    hinton_fn(pool_data, ax, max_val)

    # ax modifications
    ax.set_xlim(0 - 1, len(pool_data[0]) + 1)
    ax.set_ylim(0 - 1, len(pool_data) + 1)
    ax.tick_params(axis="x", which="major", pad=10)
    ax.tick_params(axis="y", which="major", pad=10)
    ax.set_xticks([0, len(pool_data[0]) // 4, 3 * len(pool_data[0]) // 4, len(pool_data[0]) - 1])  # type: ignore
    ax.set_xticklabels(["1", data.shape[1] // 4, 3 * data.shape[1] // 4, data.shape[1]])  # type: ignore
    # ax.set_xlabel("Time")
    ax.text(len(pool_data[0]) / 2, -1.75, "Time", ha="center")
    # put config in the title with small text (center aligned)
    title = title_fn(cfg)
    ax.text(len(pool_data[0]) / 2, len(pool_data) + 0.1, title, ha="center", fontsize=10)
    # y lbal (tasks)
    # ax.set_ylabel("Task")
    ax.text(-1.75, len(pool_data) / 2 - 0.5, "Task", va="center", rotation=90)
    title_pos = len(pool_data[0]), len(pool_data) / 2 - 0.5
    ax.text(*title_pos, f"{split} {metric}", va="center", rotation=90)
    yticks = [i for i in range(len(ds.info.tasks))]  # type: ignore
    y_tick_labels = ds.info.tasks[:-1] + ["ℙ"]
    # remove the two middle ticks
    yticks = yticks[: len(yticks) // 2 - 1] + yticks[len(yticks) // 2 + 1 :]
    y_tick_labels = y_tick_labels[: len(y_tick_labels) // 2 - 1] + y_tick_labels[len(y_tick_labels) // 2 + 1 :]
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_tick_labels)
    plt.tight_layout()
    plt.savefig(f"{path}/{split}_{metric}_hinton.svg")


def hinton_fn(data, ax, scale: float = 1.0):  # <- Hinton atomic
    """Plot a matrix of data in a hinton diagram."""
    for (y, x), w in np.ndenumerate(data):
        c = bg if w < 0 else fg  # color
        s = np.sqrt(np.abs(w) / scale) * 0.8
        if s == jnp.nan:
            s = 0
        s = jnp.clip(s, 0, 0.9).item()
        ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor=c, edgecolor=fg))

    # ax modifications
    [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale_view()
    ax.set_aspect("equal", "box")


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

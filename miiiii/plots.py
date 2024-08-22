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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Sequence
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import os

# %% Constants and configurations
fg = "black"
bg = "white"
plt.rcParams["font.family"] = "Monospace"


# %% Hinton plots
def hinton_metric(metrics, metric, ds):
    data = metrics.get(metric)
    fig, ax = plt.subplots(figsize=(12, 4))
    pool_data = mi.stats.horizontal_mean_pooling(data)
    hinton_fn(pool_data, ax)

    # ax modifications
    ax.set_xlim(0 - 1, len(pool_data[0]) + 1)
    ax.set_ylim(0 - 1, len(pool_data) + 1)
    ax.tick_params(axis="x", which="major", pad=10)
    ax.tick_params(axis="y", which="major", pad=10)
    ax.set_xticks([0, len(pool_data[0]) - 1])
    ax.set_xticklabels(["1", data.shape[1]])
    # title_pos = len(pool_data[0]) + 0.5, len(pool_data) / 2
    # ax.text(*title_pos, f"{metric} in time", ha="center", va="center", rotation=270)
    ax.set_yticks([i for i in range(len(ds.info.tasks))])  # type: ignore
    ax.set_yticklabels(ds.info.tasks[:-1] + ["ℙ"])
    plt.tight_layout()
    plt.savefig(f"figs/{metric}.svg")


def hinton_fn(data, ax):  # <- Hinton atomic
    """Plot a matrix of data in a hinton diagram."""
    scale = jnp.max(jnp.abs(data)) / 0.8
    for (y, x), w in np.ndenumerate(data):
        c = bg if w < 0 else fg  # color
        s = np.sqrt(np.abs(w) / scale)  # size
        ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor=c, edgecolor=fg))
    hinton_ax(ax)


def hinton_ax(ax):
    # remove ticks and labels
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale_view()
    ax.set_aspect("equal", "box")


# %% Polar plots
def polar_fn(ax, data):
    """Plot a polar diagram of data."""
    ax.plot(data, np.ones_like(data), "o", color=fg)


# %% Curve plots
def curve_fn(ax, data):
    """Plot a curve of data."""
    ax.plot(data, color=fg)


# %% Figures


# fig functions
def syrkis_plot(matrix, cfg, metric, ds):
    cols = matrix.shape[1] * 3
    X = matrix[: (matrix.shape[0] // cols) * cols]
    _I = jnp.identity(cols).repeat(X.shape[0] // cols, axis=0) / (X.shape[0] // cols)
    matrix = (_I[:, :, None] * X[:, None, :]).sum(axis=0).T
    one, two = matrix.shape
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95)
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())  # type: ignore
    ax.yaxis.set_major_locator(plt.NullLocator())  # type: ignore
    matrix = matrix / 0.2
    for (y, x), w in np.ndenumerate(matrix):
        s = float(jnp.sqrt(w))
        rect = plt.Rectangle(  # type: ignore
            [x - s / 2, y - s / 2],  # type: ignore
            s,
            s,
            facecolor="black",
            edgecolor="black",  # type: ignore
        )
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.set_xlim(-1, two + 1)  # important for symmetry
    ax.set_ylim(-1, one + 1)  # important for symmetry
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", which="major", pad=10)
    ax.tick_params(axis="y", which="major", pad=10)
    ax.set_xticks(
        [0, two - 1],
    )  # type: ignore
    ax.set_xticklabels(
        [1, cfg.epochs],  # type: ignore
    )  # type: ignore
    ax.set_yticks([i for i in range(len(ds.info.tasks))])  # type: ignore
    ax.set_yticklabels(ds.info.tasks[:-1] + ["ℙ"])
    ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(path_fn(cfg, metric))
    # plt.close()


def polar_plot(
    ps: Sequence[Sequence] | Sequence | np.ndarray,
    f_name: Sequence[str] | str | None = None,
    ax=None,
):
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
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        subplot_kw=dict(polar=True),
        figsize=(n_cols * 5, n_rows * 5),
        dpi=100,
    )
    for ax, fname, seqs in zip(axes.flat, fnames, seqs):  # type: ignore
        polar_plot(seqs, fname, ax=ax)
    # tight
    plt.tight_layout()
    plt.savefig(f"paper/figs/{f_name}.svg") if f_name else plt.show()  # test


def curve_plot(
    curves,
    conf,
    params,
    info=dict(
        title="training curves",
        xlabel="Epoch",
        ylabel="Focal loss",
        legend=["Training", "Validation"],
    ),
):
    conf = conf.__dict__
    conf["n_params"] = sum([p.size for p in tree_flatten(params)[0]])
    fig, ax = init_curve_plot(info, conf)
    for i, curve in enumerate(curves.T):  # transpose bcs of jax.lax.scan
        ax.plot(curve, c=fg, lw=2, ls="--" if i > 0 else "-")
    # make x-axis log
    ax.set_xscale("log")
    # add info key, val pairs to right side of plot (centered verticall on right axis) (relative to len(info))
    ignore = ["in_d", "out_d", "block"]
    conf = {k: v for k, v in conf.items() if k not in ignore}
    for i, (k, v) in enumerate(conf.items()):
        k = k[2:] if k.startswith("n_") else k
        ax.text(
            1.01,
            0.93 - i / (len(conf)),
            # if v is a number use scientific notation
            f"{k} : {v}",
            transform=ax.transAxes,
            color=fg,
        )
    ax.legend(info["legend"], frameon=False, labelcolor=fg)
    # make fname contain conf


############################################
# INITIALIZATION
############################################


def init_curve_plot(info, conf):
    # title is block with large first letter
    title = conf["block"][0].upper() + conf["block"][1:] + " " + info["title"]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg)
    # make fg white
    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg)
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.set_title(title, color=fg)
    ax.set_xlabel(info["xlabel"], color=fg)
    ax.set_ylabel(info["ylabel"], color=fg)
    # font color of legend should also be fg
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    [spine.set_edgecolor(fg) for spine in ax.spines.values()]
    # ax y range from 0 to what it is
    return fig, ax


###############################################################
# %% helpers


def path_fn(cfg, kind):
    """Creates a path for saving figures"""
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if not os.path.exists(os.path.join("figs", kind)):
        os.mkdir(os.path.join("figs", kind))
    return os.path.join("figs", kind, fname_fn(cfg))


def fname_fn(cfg):
    """Creates a filename for saving figures"""
    return f"{cfg.base}_{cfg.n}_{cfg.l2}_{cfg.dropout}.svg"


def ax_fn(ax):
    """Cleans default matplotlib stuff"""
    ax.patch.set_facecolor(bg)
    [ax.spines[loc].set_visible(False) for loc in ["top", "right", "left", "bottom"]]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

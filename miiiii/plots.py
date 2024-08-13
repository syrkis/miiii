# plot.py
#    miii plots
# by: Noah Syrkis

# imports
import miiiii as mi
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from jax.tree_util import tree_flatten


# constants
cols = {True: "black", False: "white"}
fg = "black"
bg = "white"
marks = ["o", "o", " ", "o"]
plt.rcParams["font.family"] = "Monospace"


def fname_fn(conf, fname):
    return (
        "-".join([f"{k}_{v}" for k, v in conf.items() if k not in ["in_d", "out_d"]])
        + f"_{fname}".replace(" ", "_").lower()
    )


# functions
def syrkis_plot(matrix, cfg, metric, x_scale="linear"):
    one, two = matrix.shape
    shape = (one / min(one, two), two / min(one, two))
    fig, ax = plt.subplots(figsize=(shape[0] * 6, shape[1] * 12), dpi=100)
    ax.patch.set_facecolor(bg)
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())  # type: ignore
    ax.yaxis.set_major_locator(plt.NullLocator())  # type: ignore
    for (x, y), w in np.ndenumerate(matrix):
        s = np.sqrt(w)
        # is last fg is blue
        # c = mi.utils.blue if y == 0 else fg
        c = fg
        rect = plt.Rectangle([x - s / 2, y - s / 2], s, s, facecolor=c, edgecolor=c)  # type: ignore
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    # ax.set_title(metric)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[0], step=cfg.epochs // 20))  # type: ignore
    # set first and last y ticks to 0 and 1
    # ax.set_yticks(["is even", "is prime"])
    plt.tight_layout()
    fname = "syrkis_" + fname_fn(cfg, metric)
    plt.savefig(f"paper/figs/{fname}.svg")


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
    fname = fname_fn(conf, "curves")
    if darkdetect.isLight():
        plt.savefig(f"figs/{fname}.pdf", dpi=100)


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

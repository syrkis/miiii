# plot.py
#    miii plots
# by: Noah Syrkis

# imports
import plotly as py
import pandas as pd
import jax.numpy as jnp
import numpy as np
import darkdetect
import matplotlib.pyplot as plt
from oeis import A000040
from functools import partial
from hilbert import encode, decode
from jax.tree_util import tree_flatten


# constants
cols = {True: "black", False: "white"}
ink = "black" if darkdetect.isLight() else "white"
bg = "white" if darkdetect.isLight() else "black"
marks = ["o", ".", "s", "D", "v", "^", "<", ">", "1", "2", "3", "4"]
plt.rcParams["font.family"] = "Monospace"


# functions
def polar_plot(vector, conf, info, fname, offset=0):  # maps v to a polar plot
    _, ax = init_polar_plot()
    cats = jnp.unique(vector)[jnp.unique(vector) > 0]
    conf = {
        k[2:] if k.startswith("n_") else k: v
        for k, v in conf.items()
        if k not in ["in_d", "out_d", "block"]
    }
    ax.set_title(" | ".join([f"{k} : {v}" for k, v in conf.items()]), color=ink, pad=25)
    # add padding to tile
    for cat, m in zip(cats, marks):
        idxs = jnp.where(vector == cat)[0] + 2 + offset
        # size = jnp.sqrt(idxs) / jnp.log(idxs)
        ax.scatter(idxs, idxs, c=ink, marker=m)
    fname = fname if fname.endswith(".pdf") else f"{fname}.pdf"
    ax.set_xlabel("    ".join([f"{k} : {v:.3f}" for k, v in info.items()]), color=ink)
    plt.savefig(f"figs/{fname}", dpi=100)


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
    conf["n_params"] = sum([p.size for p in tree_flatten(params)[0]])
    fig, ax = init_curve_plot(info, conf)
    for i, curve in enumerate(curves.T):  # transpose bcs of jax.lax.scan
        ax.plot(curve, c=ink, lw=2, ls="--" if i > 0 else "-")
    # add info key, val pairs to right side of plot (centered verticall on right axis) (relative to len(info))
    ignore = ["in_d", "out_d", "block"]
    conf = {k: v for k, v in conf.items() if k not in ignore}
    for i, (k, v) in enumerate(conf.items()):
        k = k[2:] if k.startswith("n_") else k
        ax.text(
            1.01,
            0.93 - i / (len(conf)),
            # if v is a number use scientific notation
            f"{k}: {v}",
            transform=ax.transAxes,
            color=ink,
        )
    ax.legend(info["legend"], frameon=False, labelcolor=ink)
    plt.savefig(f"figs/curves.pdf")


############################################
# INITIALIZATION
############################################


def init_curve_plot(info, conf):
    # title is block with large first letter
    title = conf["block"][0].upper() + conf["block"][1:] + " " + info["title"]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg)
    # make ink white
    ax.tick_params(axis="x", colors=ink)
    ax.tick_params(axis="y", colors=ink)
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.set_title(title, color=ink)
    ax.set_xlabel(info["xlabel"], color=ink)
    ax.set_ylabel(info["ylabel"], color=ink)
    # font color of legend should also be ink
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    [spine.set_edgecolor(ink) for spine in ax.spines.values()]
    # ax y range from 0 to what it is
    return fig, ax


def init_polar_plot():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(12, 12))
    fig.patch.set_facecolor(bg)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_facecolor(bg)
    [spine.set_edgecolor(ink) for spine in ax.spines.values()]
    # plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    from datum import data_fn
    from numbs import base_n

    y = data_fn(A000040, 2**16, partial(base_n, n=2))[1]
    polar_plot(y, "primes")  # plot of primes

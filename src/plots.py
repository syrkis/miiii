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
from sklearn.metrics import f1_score, confusion_matrix

if __name__ == "__main__":
    from utils import alpha_fn
else:
    from .utils import alpha_fn


# constants
cols = {True: "black", False: "white"}
ink = "black" if darkdetect.isLight() else "white"
bg = "white" if darkdetect.isLight() else "black"
marks = ["o", "o", " ", "o"]
plt.rcParams["font.family"] = "Monospace"


# functions
def polar_plot(gold, pred, conf, fname, offset=0):  # maps v to a polar plot
    conf = conf.__dict__
    _, ax = init_polar_plot()
    tp, tn = gold + pred == 2, gold + pred == 0
    fp, fn = gold - pred == -1, gold - pred == 1
    f1 = f1_score(gold, pred)
    con = confusion_matrix(gold, pred)
    tp_rate = con[0, 0] / (con[0, 0] + con[0, 1])
    tn_rate = con[1, 1] / (con[1, 0] + con[1, 1])
    fp_rate = con[0, 1] / (con[0, 0] + con[0, 1])
    fn_rate = con[1, 0] / (con[1, 0] + con[1, 1])
    info = {"f1": f1, "tp": tp_rate, "tn": tn_rate, "fp": fp_rate, "fn": fn_rate}
    vectors = {"tp": tp, "fn": fn}
    conf = {
        k[2:] if k.startswith("n_") else k: v
        for k, v in conf.items()
        if k not in ["in_d", "out_d", "block"]
    }
    # add padding to tile
    for idx, (cat, vector) in enumerate(vectors.items()):
        # if cat == fn Make it an empty circle
        idxs = jnp.where(vector)[0] + 2 + offset
        ax.scatter(
            idxs,
            idxs,
            marker=marks[idx],
            # s=10 + 4 * idx,
            label=cat,
            facecolors=bg if cat == "fn" else ink,
            edgecolors=ink,
        )
    fname = fname if fname.endswith(".pdf") else f"{fname}.pdf"
    xlabel = dict(**info, **conf)
    xlabel["alpha"] = alpha_fn(conf["n"] // 2).item()
    # delete digits from xlabel
    xlabel.pop("digits")
    # join every fifth element with a newline
    v_fn = lambda v: f"{v:.3f}" if isinstance(v, float) else v
    xlabel = "\n\n\n\n".join(
        [
            "    ".join([f"{k} : {v_fn(v)}" for k, v in xlabel.items()][i : i + 5])
            for i in range(0, len(xlabel), 5)
        ]
    )
    ax.set_xlabel(xlabel, color=ink)
    if darkdetect.isLight():
        plt.savefig(f"figs/{fname}", dpi=100)
    else:
        # invert image and svae
        plt.savefig(f"figs/{fname}", dpi=100, facecolor=bg, edgecolor=ink)


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
            f"{k} : {v}",
            transform=ax.transAxes,
            color=ink,
        )
    ax.legend(info["legend"], frameon=False, labelcolor=ink)
    if darkdetect.isLight():
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

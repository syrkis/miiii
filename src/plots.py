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


# constants
cols = {True: "black", False: "white"}
ink = "black" if darkdetect.isLight() else "white"
bg = "white" if darkdetect.isLight() else "black"
marks = ["x", "o", "o", "s", "D", "v", "^", "<", ">", "1", "2", "3", "4"]


# functions
def polar_fn(vector, fname, offset=0):  # maps v to a polar plot
    fig, ax = init_polar_plot()
    cats = jnp.unique(vector)[jnp.unique(vector) > 0]
    for cat, m in zip(cats, marks):
        idxs = jnp.where(vector == cat)[0] + 2 + offset
        size = jnp.sqrt(idxs) / jnp.log(idxs)
        ax.scatter(idxs, idxs, c=ink, marker=m)
    plt.savefig(f"figs/{fname}.pdf", dpi=300)


def init_polar_plot():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(18, 18))
    fig.patch.set_facecolor(bg)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # set scatter colors to black
    ax.set_facecolor(bg)
    [spine.set_edgecolor(ink) for spine in ax.spines.values()]
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    from datum import data_fn
    from numbs import base_n

    y = data_fn(A000040, 2**16, partial(base_n, n=2))[1]
    polar_fn(y, "primes")  # plot of primes

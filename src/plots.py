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


def polar_fn(v):  # maps v to a polar plot
    # there are three groups of dots. dots from train set. test set that are correctly classified. test set that are incorrectly classified
    # test set dots are round, and filled if correct, and empty if not.
    # train set dots are triangles, and filled if correct, and empty if not.
    primes = jnp.array(A000040[1 : v.size + 1])
    primes = primes[primes < v.max()]
    col = {True: "black", False: "white"}
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(14, 14))
    ax.scatter(
        v[primes],
        v[primes],
        color=col[darkdetect.isLight()],
        s=jnp.sqrt(v[primes]) / jnp.log(v.size),
        marker="o",
    )
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_facecolor(col[darkdetect.isDark()])
    fig.patch.set_facecolor(col[darkdetect.isDark()])
    [spine.set_edgecolor(col[darkdetect.isLight()]) for spine in ax.spines.values()]
    plt.tight_layout()
    plt.savefig("figs/polar.pdf", dpi=300)


def hilbert_fn(v):  # maps v to a hilbert curve
    pass


if __name__ == "__main__":
    v = jnp.arange(2**16)
    polar_fn(v)

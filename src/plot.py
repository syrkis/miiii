# plot.py
#    miii plots
# by: Noah Syrkis

# imports
import plotly as py
import pandas as pd
import jax.numpy as jnp
import numpy as np
import darkdetect
from datetime import date, timedelta
import tikz
import matplotlib.pyplot as plt


def polar_fn(v):  # maps v to a polar plot
    ink = "black" if darkdetect.isLight() else "white"
    bg = "white" if darkdetect.isLight() else "black"
    scale = jnp.log(v.size)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(14, 14))
    ax.grid(False)
    # set color to ink
    ax.scatter(v, v, color=ink, s=10 / scale)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # set colors to ink and bg
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    plt.rcParams["text.color"] = ink
    # tight layout
    plt.tight_layout()
    plt.show()


def hilbert_fn(v):  # maps v to a hilbert curve
    pass


if __name__ == "__main__":
    v = jnp.arange(1000)
    polar_fn(v)

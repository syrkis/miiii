# plot.py
#    miii plots
# by: Noah Syrkis

# imports
import plotly as py
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tikz
import matplotlib.pyplot as plt


def polar_fn(v):  # maps v to a polar plot
    # map each point n in v to (n, n * theta)
    theta = 2 * np.pi / len(v)
    coords = [(n, n * theta) for n in v]
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(coords)
    plt.show()


def hilbert_fn(v):  # maps v to a hilbert curve
    pass


# tikz functions
def plot_fn():
    pic = tikz.Picture()
    # mlp in -> activation -> mlp out
    pic.draw("(-1, 0) rectangle (1, 1)")
    pic.draw("(-1, 0.5) -- (1, 0.5)", "dashed")
    pic.draw("(-1, 0.5) circle (0.5)")
    pic.draw("(-1, 0.5) -- (-1, 1.5)", "dashed")
    pic.draw("(1, 0.5) -- (1, 1.5)", "dashed")
    pic.draw("(-1, 1.5) rectangle (1, 2.5)")
    pic.draw("(-1, 2) -- (1, 2)", "dashed")
    pic.draw("(0, 2) circle (0.5)")
    pic.draw("(0, 2) -- (0, 3)", "dashed")
    pic.draw("(0, 3) rectangle (1, 4)")
    pic.draw("(0, 3.5) -- (1, 3.5)", "dashed")
    pic.draw("(0, 3.5) circle (0.5)")
    pic.draw("(0, 3.5) -- (0, 4.5)", "dashed")
    pic.draw("(0, 4.5) rectangle (1, 5.5)")
    # save
    pic.write("mlp.tikz")
    # save as png


def main():
    plot_fn()


if __name__ == "__main__":
    main()

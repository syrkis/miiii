# %% main.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import numpy as np
from jax import random
from typing import Sequence
import matplotlib.pyplot as plt
from oeis import A000040 as primes

# %% Exploring and plotting the data
cfg, rng = mi.utils.get_conf(), random.PRNGKey(seed := 0)
rng, key = random.split(rng)
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, key)


# fig, ax = plt.subplots(figsize=ds.train.y.shape, dpi=100)
# sns.heatmap(ds.train.y.T, ax=ax, cmap="Greys", cbar=False)


# %% Polar plot
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
    # close plot
    plt.close()


fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
seqs = [
    range(1, 1024),
    range(0, 1024, 6),
    [range(0, 1024, 2), range(0, 1024, 5)],
    np.array(primes[1025:2049]),
]
small_multiples(fnames[:3], seqs[:3], "polar_nats_and_sixes", 1, 3)
polar_plot(seqs[-1], "polar_primes")

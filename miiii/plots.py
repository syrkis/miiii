# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import miiii as mi
import esch
import os
from typing import Sequence

import jax.numpy as jnp
from jax.numpy import fft
import matplotlib.pyplot as plt

import numpy as np
from jax import Array


# %% Constants and configurations
plt.rcParams["font.family"] = "Monospace"
plt.rcParams["mathtext.fontset"] = "cm"

# figs dir is in ../paper/figs relative to THIS file
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../paper/figs")


# %% functions
def log_axis_array(arr, length=100):
    # array is (time, task)
    # out put will be (task, 100)
    # we get to 100 by doing [:: cfg.epochs // 100]
    # The x-axis should be log-scaled in time
    arr = arr.T  # make it (task, time)
    # step_size = arr.shape[1] // 100
    # create log-spaced indices
    log_indices = np.logspace(0, np.log10(arr.shape[1]), length, dtype=int) - 1
    log_indices = np.clip(log_indices, 0, arr.shape[1] - 1)
    arr = arr[:, log_indices]
    return arr


def plot_run(
    metrics,
    ds: mi.tasks.Dataset,
    cfg: mi.utils.Conf,
    hash,
    activations=None,
    font_size=12,
    log_axis=False,
    log_scale=False,
):
    os.makedirs(os.path.join(FIGS_DIR, hash), exist_ok=True)

    # training and final plots
    splits = ["train", "valid"]
    _metrics = ["loss", "acc"]
    for s in splits:
        for m in _metrics:
            plot_training(
                getattr(getattr(metrics, s), m),
                s,
                m,
                cfg,
                hash,
                font_size=font_size,
                log_axis=log_axis,
                log_scale=log_scale,
            )

    # plot exploratory plots
    # attention weights. SVD. etc.


def fourier_analysis(matrix):
    # Compute 2D FFT
    fft_2d = fft.rfft2(matrix.T).T
    magnitude_spectrum = jnp.abs(fft_2d)

    # Get significant frequencies
    # freq_tokens = fft.fftfreq(matrix.shape[0])
    # freq_embeds = fft.fftfreq(matrix.shape[1])

    # Center everything
    magnitude_spectrum_centered = fft.fftshift(magnitude_spectrum)
    # freq_tokens_centered = fft.fftshift(freq_tokens)
    # freq_embeds_centered = fft.fftshift(freq_embeds)

    # frequency activations
    freq_activations = jnp.linalg.norm(magnitude_spectrum_centered, axis=1)

    significant_freqs = freq_activations > freq_activations.mean()

    return magnitude_spectrum_centered, freq_activations, significant_freqs


def plot_training(metric, split, name, cfg: mi.utils.Conf, hash, font_size=1.0, log_axis=False, log_scale=False):
    path = os.path.join(FIGS_DIR, hash, f"{name}_{split}_training.svg")
    if log_axis:
        data = log_axis_array(metric)[:, 10:]
    else:
        data = metric[:: cfg.epochs // 100].T
    if log_scale:
        data = np.log10(data + 1e-8) + 1e-8
    left = esch.EdgeConfig(label="Task", show_on="first")

    ticks = [(i, str(prime.item())) for i, prime in enumerate(task.primes) if i % 2 == 0]
    right = esch.EdgeConfig(ticks=ticks, show_on="all")  # type: ignore
    bottom = esch.EdgeConfig(
        ticks=[(0, "1"), (90 - 1, f"{cfg.epochs:g}")],
        show_on="first",
        label=f"Time ({'log' if log_axis else 'linear'})",
    )
    name = f"{split.capitalize()} Accuracy" if name == "acc" else f"{split.capitalize()} Cross Entropy"
    top = esch.EdgeConfig(label=name.capitalize(), show_on="all")
    edge = esch.EdgeConfigs(right=right, top=top, left=left, bottom=bottom)
    esch.plot(data, path=path, edge=edge, font_size=font_size)


def plot_final(metric, split, name, cfg, task, hash):
    path = os.path.join(FIGS_DIR, hash, f"{name}_{split}_final.svg")
    data = metric[-1][None, :]
    ticks = [(i, str(prime.item())) for i, prime in enumerate(task.primes) if i % 2 == 0]
    bottom = esch.EdgeConfig(ticks=ticks, show_on="first")  # type: ignore
    edge = esch.EdgeConfigs(bottom=bottom)
    esch.plot(data, path=path, edge=edge)


def plot_svd(w, name, cfg):
    U, S, V = jnp.linalg.svd(w)


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
    plt.savefig(f"{FIGS_DIR}/{f_name}.svg") if f_name else plt.show()
    if ax_was_none:
        plt.close()


def small_multiples(fnames, seqs, f_name, n_rows=2, n_cols=2):
    assert len(fnames) == len(seqs) and len(fnames) >= n_rows * n_cols, (
        "fnames and seqs must be the same length and n_rows * n_cols"
    )
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(n_cols * 5, n_rows * 5), dpi=100)
    for ax, fname, seqs in zip(axes.flat, fnames, seqs):  # type: ignore
        polar_plot(seqs, fname, ax=ax)
    # tight
    # plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/{f_name}.svg") if f_name else plt.show()  # test


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
            ax.plot(curve, c="black", lw=2, ls="--" if i > 0 else "-")
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
    fig.patch.set_facecolor("white")
    # make fg white
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.set_facecolor("white")
    ax.grid(False)
    # ax.set_title(title, color=fg)
    # ax.set_xlabel(info["xlabel"], color=fg)
    # ax.set_ylabel(info["ylabel"], color=fg)
    # font color of legend should also be fg
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    [spine.set_edgecolor("black") for spine in ax.spines.values()]
    # ax y range from 0 to what it is
    return fig, ax


def horizontal_mean_pooling(x: Array, width: int = 3) -> Array:
    """Rolling mean array. Shrink to be rows x rows * width."""
    x = x[:, : (x.shape[1] // (x.shape[0] * width)) * (x.shape[0] * width)]
    i = jnp.eye(x.shape[0] * width).repeat(x.shape[1] // (x.shape[0] * width), axis=-1)
    z = (x[:, None, :] * i[None, :, :]).sum(axis=-1)
    return z / (x.shape[1] // (x.shape[0] * width))

# %% plot.py
#    miii plots
# by: Noah Syrkis


# %% imports
import miiii as mi
import esch
from einops import rearrange
import os

import jax.numpy as jnp
from jax.numpy import fft
import matplotlib.pyplot as plt

import numpy as np
from jax import Array

folder = "/Users/nobr/desk/s3/miiii"


# %% plot y
def plot_y(cfg, ds):
    arr = np.array(rearrange(ds.y, "(n p) t -> t n p", n=cfg.p, p=cfg.p))[-1][None, ...] / cfg.p
    e = esch.Drawing(h=cfg.p - 1, w=cfg.p - 1, row=1, col=arr.shape[0])
    esch.grid_fn(arr, e, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_y.svg")


# %% plot x
def plot_x(cfg, ds: mi.types.Dataset):
    arr = np.array(rearrange(ds.x[:, :2], "(n p) d -> n d p", n=cfg.p, p=cfg.p)) + 0.5
    e = esch.Drawing(w=cfg.p - 1, h=2 - 1, col=cfg.p, row=1)
    esch.grid_fn(arr / arr.max(), e, shape="square")
    esch.save(e.dwg, f"{folder}/{cfg.p}_x.svg")

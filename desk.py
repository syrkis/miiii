# %%
import mlxp
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oeis import oeis
import seaborn as sns
from omegaconf import DictConfig
from einops import rearrange
import esch

# Constants
cfg = DictConfig({"p": 113, "tick": 512, "l2": 0.33, "lamb": 0.5})
reader = mlxp.Reader("./logs/", refresh=True)


# %%
def primes_fn(p):  # return array of primes up to and including p
    return np.array(oeis["A000040"][1:p])[np.array(oeis["A000040"][1:p]) <= p]


def cfg_name(cfg):
    return f"p{cfg.p}_tick{cfg.tick}_l2{cfg.l2}_lamb{cfg.lamb}"


def cfg_to_df(cfg):
    query: str = f"info.status == 'COMPLETE' & config.p == {cfg.p} & config.tick == {cfg.tick} & config.l2 == {cfg.l2} & config.lamb == {cfg.lamb}"
    return pd.DataFrame(reader.filter(query_string=query)).iloc[-1]


def log_index(length):
    return np.unique(np.round(np.logspace(0, np.log10(length - 1), num=min(length, 512)))).astype(int)


def plot_run(cfg):
    sample = cfg_to_df(cfg)
    dir_name = cfg_name(cfg)
    os.makedirs(f"figs/{dir_name}", exist_ok=True)

    [plot_curves(sample, key, f"figs/{dir_name}") for key in "train_cce valid_cce train_acc valid_acc".split()]
    plot_neu(sample, f"figs/{dir_name}")


def plot_curves(sample, key, path):
    arr = np.array(sample["scope." + key])[log_index(cfg.tick)][:, ::3] ** 0.5
    e = esch.draw("w n h", arr / (arr.max() + 0.01))  # -3, keepdims=True))
    e.save(f"{path}/curves_{key}.svg")


def plot_neu(sample, path):
    neu = sample["artifact.pickle."]["neu.pkl"].load()
    tmp = neu[::3, ::30, ::16]
    e = esch.draw("n t c h w", tmp / (tmp.max((-2, -1), keepdims=True) + 0.1))
    e.save(f"{path}/neu.svg")

    # plot fft
    fft = np.abs(np.fft.fft2(neu))
    tmp = fft[::3, ::30, ::16]
    e = esch.draw("n t c h w", tmp / (tmp.max((-2, -1), keepdims=True) + 0.1))
    e.save(f"{path}/fft.svg")

    # plot fft curve
    mu, sigma = fft.mean((-2, -1), keepdims=True), fft.std((-2, -1), keepdims=True)
    tmp = (fft > mu + sigma * 2).mean((-2, -1))
    e = esch.draw("n w h", (tmp / (tmp.max(0, keepdims=True)))[:, log_index(tmp.shape[1])][::3])
    e.save(f"{path}/fft_curve.svg")
    plt.plot(tmp.sum(-1).T, label=range(7))
    plt.show()


# %%
plot_run(DictConfig({**cfg, "lamb": 0.5}))
plot_run(DictConfig({**cfg, "lamb": 0.0}))

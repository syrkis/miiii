# %%
import mlxp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
from oeis import oeis
from omegaconf import OmegaConf


def primes_fn(p):  # return array of primes up to and including p
    return jnp.array(oeis["A000040"][1:p])[jnp.array(oeis["A000040"][1:p]) <= p]


# %%
cfg = OmegaConf.load("conf/config.yaml")
reader = mlxp.Reader("./logs/", refresh=True)
query = f"info.status == 'COMPLETE' & config.p == 113 & config.epochs == {cfg.epochs} & config.tick == {cfg.tick} & config.device == 'hpc'"
df = pd.DataFrame(reader.filter(query_string=query))


df
# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(np.array(df["scope.train_acc"].to_list())[-1], alpha=0.5, label=primes_fn(cfg.p))
axes[1].plot(np.array(df["scope.valid_acc"].to_list())[-1], alpha=0.5, label=primes_fn(cfg.p))
for ax in axes:
    ax.set_xlabel("Tick")
    ax.set_ylabel("Acc")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True)


# %%
def plot_last(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, key in zip(axes.flatten(), ["train_acc", "train_cce", "valid_acc", "valid_cce"]):
        ax.plot(df.iloc[-1]["scope." + key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(["Last"])


plot_last(df)


# %%
def plot_mean(df, idx):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, key in zip(axes.flatten(), ["train_acc", "train_cce", "valid_acc", "valid_cce"]):
        ax.plot(df.iloc[idx]["scope." + key].mean(-1))
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(["Mean"])


plot_mean(df, idx=-1)

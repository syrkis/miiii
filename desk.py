# %%
import mlxp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oeis import oeis
import seaborn as sns
from omegaconf import OmegaConf


def primes_fn(p):  # return array of primes up to and including p
    return np.array(oeis["A000040"][1:p])[np.array(oeis["A000040"][1:p]) <= p]


# %%
cfg = OmegaConf.load("conf/config.yaml")
reader = mlxp.Reader("./logs/", refresh=True)
query: str = "info.status == 'COMPLETE' & config.p == 83 & config.tick == 128"
df = pd.DataFrame(reader.filter(query_string=query))
# df = df.loc[df["info.hostname"].map(lambda x: x.endswith("hpc.itu.dk"))]  # use remote


# %%
def plot_metrics(sample, idx):
    prime = sample["config.p"]
    primes = primes_fn(prime)
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    axes[0].plot(np.array(sample["train.loss"])[:, idx])
    axes[1].plot(np.array(sample["scope.train_cce"])[:, idx], alpha=0.5, label=primes.tolist())
    axes[2].plot(np.array(sample["scope.valid_cce"])[:, idx], alpha=0.5, label=primes.tolist())
    axes[3].plot(np.array(sample["scope.train_acc"])[:, idx], alpha=0.5, label=primes.tolist())
    axes[4].plot(np.array(sample["scope.valid_acc"])[:, idx], alpha=0.5, label=primes.tolist())
    for jdx, ax in enumerate(axes):
        ax.set_title("train_loss train_cce valid_cce train_acc valid_acc".split()[jdx])
        ax.set_ylabel(str(p) + " or larger" if jdx == 0 else "")
        ax.set_xscale("log")
        if jdx > 0:
            ax.legend()
        ax.set_xticklabels([])
        ax.grid(True)


sample = df.iloc[-1]
primes = primes_fn(sample["config.p"])
for idx in range(np.array(df["train.loss"].iloc[-1]).shape[1]):
    plot_metrics(sample, idx)


# %%
arr = sample["artifact.pickle."]["neu.pkl"].load()


def plot_neu(arr, idx):
    fig, axes = plt.subplots(8, 16, figsize=(10, 5))
    for jdx, ax in enumerate(axes.flatten()):
        sns.heatmap(arr[idx, -1, jdx], ax=ax, cmap="grey", cbar=False)
        ax.axis("off")
    plt.show()


for idx, p in enumerate(primes):
    plot_neu(arr, idx)


# %%
def plot_omega(arr):
    fig, axes = plt.subplots(3, 3, figsize=(20, 8))
    for idx, ax in enumerate(axes.flatten()):
        tmp = arr[idx]
        fft = np.abs(np.fft.fft2(tmp))[..., 1:, 1:]
        mu, sigma = fft.mean((-2, -1), keepdims=True), fft.std((-2, -1), keepdims=True)
        data = (fft > (mu + 5 * sigma)).sum((-2, -1))
        # Create coordinate grid
        y_dim, x_dim = data.T.shape
        x = np.arange(x_dim + 1)
        y = np.arange(y_dim + 1)
        pcm = ax.pcolormesh(x, y, data.T, cmap="grey", shading="auto")
        ax.axis("off")

    plt.show()


plot_omega(arr)

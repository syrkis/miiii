# %%
import mlxp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oeis import oeis
import seaborn as sns
from omegaconf import OmegaConf
from einops import rearrange


def primes_fn(p):  # return array of primes up to and including p
    return np.array(oeis["A000040"][1:p])[np.array(oeis["A000040"][1:p]) <= p]


# %%
cfg = OmegaConf.load("conf/config.yaml")
reader = mlxp.Reader("./logs/", refresh=True)
query: str = "info.status == 'COMPLETE' & config.p == 83 & config.tick == 128 & config.l2 == 0.1"
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
        ax.set_ylabel("mask " + str(idx) if jdx == 0 else "")
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
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for jdx, ax in enumerate(axes.flatten()):
        sns.heatmap(arr[idx, -1, jdx], ax=ax, cmap="grey", cbar=False)
        ax.axis("off")
    plt.show()


for idx in range(np.array(df["train.loss"].iloc[-1]).shape[1]):
    plot_neu(arr, idx)


# %%
def plot_omega(arr):
    fig, axes = plt.subplots(4, figsize=(20, 40))

    for idx, ax in enumerate(axes.flatten()):
        tmp = rearrange(arr[idx], "a b ... -> b a ...")
        print(tmp.shape)
        fft = np.abs(np.fft.fft2(tmp))[..., 1:, 1:]
        mu, sigma = fft.mean((-4, -3, -2, -1), keepdims=True), fft.std((-4, -3, -2, -1), keepdims=True)
        data = (fft > (mu + 2 * sigma)).mean((-2, -1))
        ax.plot(data.sum(0))
        # sns.heatmap(data, ax=ax, cmap="grey", cbar=False)
        # logarithmic_data = np.logspace(10, np.log(data.shape[1]), data.shape[1] + 1)
        # ax.pcolormesh(logarithmic_data, np.arange(data.shape[0] + 1), data, shading="flat", cmap="grey")
        # ax.set_xscale("log")
        # ax.set_xlim(logarithmic_data[0], logarithmic_data[-1])

    plt.show()


plot_omega(arr)
# %%

fft = np.abs(np.fft.fft2(arr))[..., 1:, 1:]
# %%
plt.imshow(fft[-1, 70, -1])

# %%
arr.shape

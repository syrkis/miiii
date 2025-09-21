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
query: str = "info.status == 'COMPLETE' & config.epochs == 1000 & config.p == 23 & config.tick == 256"
df = pd.DataFrame(reader.filter(query_string=query))
# df = df.loc[df["info.hostname"].map(lambda x: x.endswith("hpc.itu.dk"))]  # use remote

# %%

np.array(df["train.loss"].to_list()).shape
plt.plot(np.array(df["train.loss"].to_list()[-1]), label=primes_fn(23))
plt.legend()


# %%
def plot_metrics(sample, idx):
    prime = sample["config.p"]
    primes = primes_fn(prime)
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    # for idx, p in enumerate(primes):
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
for idx, p in enumerate(primes):
    plot_metrics(sample, idx)


# %%
def plot_neu(arr, idx):
    fig, axes = plt.subplots(8, 16, figsize=(20, 10))
    for jdx, ax in enumerate(axes.flatten()):
        sns.heatmap(arr[idx, ..., jdx], ax=ax, cmap="grey", cbar=False)
        ax.axis("off")
    plt.show()


arr = sample["artifact.pickle."]["127_neu.pkl"].load()
for idx, p in enumerate(primes):
    plot_neu(arr, idx)

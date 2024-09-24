# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi  # test
from jax import random, vmap, lax, nn
import jax.numpy as jnp
from functools import partial
from oeis import A000040 as primes
import seaborn as sns
import matplotlib.pyplot as plt


p = 113

# %% Training
cfg = mi.utils.cfg_fn(depth=3, lr=1e-4, heads=4, n=p**2, base=p, latent_dim=128, epochs=10_000, dropout=0.1, l2=1.0)
keys = random.split(random.PRNGKey(0))
ds = mi.prime.prime_fn(cfg, keys[0])
(params, *_), metrics = mi.train.train(keys[1], cfg, ds)
# ds = mi.prime.unsplit_fn(train_ds)


# %% Model
def block_fn(z, param):
    attn, attn_acts = mi.model.attn_fn(param.attn, z)
    ffwd, ffwd_acts = mi.model.ffwd_fn(param.ffwd, z + attn)
    return z + ffwd, (attn_acts, ffwd_acts)  # <-- return activations from inside the ffwd block


@partial(vmap, in_axes=(None, 0))
def scope_fn(params, x):
    embeds = mi.model.embed_fn(params.embeds, x)
    z, acts = lax.scan(block_fn, embeds, params.blocks)
    logits = mi.model.base_n_pos_weigh(z @ params.unbeds, cfg.base)
    return embeds, acts, logits


# %% Hinton plots
# embeds, (attn_acts, ffwd_acts), logits = scope_fn(params, ds.x)
# mi.plots.hinton_fn(attn_acts.k)

# plot_head_activations(attn_acts.k.mean(axis=0)[0])
# mi.plots.plot_head_activations(attn_acts.wei.mean(axis=0)[0])
# mi.plots.plot_head_activations(attn_acts.wei.mean(axis=0)[1])
# %%
# mi.plots.hinton_fn(attn_acts.wei.mean(axis=0)[0].mean(axis=-1))


# %%
def attention_hintons(attn_acts, layer, a, b):
    wei = attn_acts.wei
    fig, axes = plt.subplots(ncols=cfg.heads, figsize=(14, 8))
    for i, ax in enumerate(axes):
        x = wei[:, layer, i, a, b].reshape(p, p)
        mi.plots.hinton_fn(x, ax, scale=1)
        # ax.set_title(f"Head {i}")
        # ax.set_ylabel("First digit")
        # ax.set_xlabel("Second digit")
        # add y ticks at 0 and 36 with those numbers
    # add text to the right side of plot (rotated 90) giving the description hellow world
    # fig.text(1, 0.5, f"Attention from digit a to b", va="center", rotation=90)
    plt.tight_layout()
    plt.savefig(f"paper/figs/attention_layer_{layer}.svg", format="svg", bbox_inches="tight")
    plt.close()


# attention_hintons(attn_acts, 0, 1, 1)
# attention_hintons(attn_acts, 0, 0, 1)
# fig, ax = plt.subplots(figsize=(8, 8))
# mi.plots.hinton_fn(attn_acts.wei[:, 0, 1, 1, 1].reshape(p, p), ax)
# ax.set_yticks([0, p - 1])
# ax.set_xticks([0, p - 1])
# plt.savefig("paper/figs/attention_one.svg", format="svg", bbox_inches="tight")


# %%
# %%


def plot_sample_activations(embeds, attn_acts, ffwd_acts, logits, ds, i=0):
    def plot_block(acts, ax):
        mi.plots.hinton_fn(acts, ax)

    # Plot 1: Embeddings
    plt.figure(figsize=(12, 4))
    plt.title("Embeddings")
    plot_block(embeds[i], plt.gca())
    # plt.ylabel(ds.train.x[i])
    plt.yticks(range(len(ds.train.x[i])), ds.train.x[i])
    plt.gca().tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.show()

    # Plot 2: Attention Activations
    fig, axes = plt.subplots(cfg.depth, 1, figsize=(12, 4 * cfg.depth))
    fig.suptitle("Attention Activations", fontsize=16)
    for j, ax in enumerate(axes):
        plot_block(attn_acts[i][j], ax)
        ax.set_title(f"Attn Layer {j+1}")
    plt.tight_layout()
    plt.show()

    # Plot 3: Feedforward Activations
    fig, axes = plt.subplots(cfg.depth, 1, figsize=(12, 4 * cfg.depth))
    fig.suptitle("Feedforward Activations", fontsize=16)
    for j, ax in enumerate(axes):
        plot_block(ffwd_acts[i][j], ax)
        ax.set_title(f"FFWD Layer {j+1}")
    plt.tight_layout()
    plt.show()

    # Plot 4: Z
    plt.figure(figsize=(12, 4))
    plot_block(logits[i][::-1], plt.gca())  # IMPORANT: MATRIX INDEX AND HINTON PLOT have different indexing
    # the x-ticks should be ds.train.y[i]
    plt.xticks(range(len(ds.info.tasks)), ds.info.tasks[:-1] + ["ℙ"])
    # plt.gca().tick_params(axis='x', rotation=90)
    plt.yticks(range(len(ds.train.x[i])), ds.train.x[i][::-1])
    plt.tight_layout()
    plt.show()


# %%
# plot_sample_activations(embeds, attn_acts, ffwd_acts, logits, ds, 113)


# %%
mi.plots.plot_run(metrics, ds, cfg)  # type: ignore
# %%
# state, metrics = train(cfg.epochs, rng, state)
# state = train(cfg.epochs, rng, state)
# mi.utils.track_metrics(metrics, ds, cfg)
# %% Hinton metrics


# %% Polar plots
# fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
# twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
# seqs = [range(1, 1024), range(0, 1024, 2), twos_and_fives, primes[1:300]]

# mi.plots.small_multiples(fnames, seqs, "polar_nats_and_sixes", 1, 4)
# mi.plots.polar_plot(seqs[-1], "polar_primes")

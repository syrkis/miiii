# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random, nn, tree
import jax.numpy as jnp
import esch
from einops import rearrange
import matplotlib.pyplot as plt

# %% Training
cfg = mi.utils.Conf(project="miiii", prime=113, epochs=1000, lamb=2, dropout=0.1, lr=0.1)
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(cfg, keys[0])

# %%
state, output = mi.train.train(keys[1], cfg, ds, scope=True)


# %%
W_E = state.params.embeds.tok_emb
W_E.shape

# %%
W_neur = (
    state.params.embeds.tok_emb
    @ state.params.blocks.attn.v[0]
    @ state.params.blocks.attn.p[0]
    @ state.params.blocks.ffwd.w1[0]
)
W_neur.shape

# %%

# %%
W_logit = state.params.blocks.ffwd.w2[0] @ state.params.unbeds
W_logit.shape

# %% scope
apply = mi.model.apply_fn(cfg)
acts = apply(state.params, rng, ds.train.x, 0.0)
tree.map(jnp.shape, acts)

# %%  we see the attention is leaning towards the first digit (from the right)
# wei = rearrange(acts.wei, "time (a b) layer head fst snd -> time a b layer head fst snd", a=cfg.prime, b=cfg.prime)
# esch.plot(
#     rearrange(
#         rearrange(
#             output[1].wei, "time (a b) layer head fst snd -> time a b layer head fst snd", a=cfg.prime, b=cfg.prime
#         ).squeeze()[-1],
#         "a b head fst snd -> (head snd) (fst a b)",
#     ),
# )


# %%
esch.plot(
    rearrange(output[1].wei[-1], "(a b) layer head fst snd -> a b layer head fst snd", a=cfg.prime, b=cfg.prime)[
        :, :, 0, 3, 0, 0
    ],
    xlabel="First digit (a)",
    ylabel="Second digit (b)",
    xticks=[(0, str(0)), (cfg.prime - 1, str(cfg.prime - 1))],
    yticks=[(0, str(0)), (cfg.prime - 1, str(cfg.prime - 1))],
)


# %%  Neural activations (first five mlp neurons)
esch.plot(rearrange(output[1].logits[-1][:, :1000], "(a b) neuron -> neuron a b", a=cfg.prime, b=cfg.prime)[100])  #

# %% Logging stuff
U, S, V = jnp.linalg.svd(W_E)

# %%
thresh = jnp.where((S / S.sum()).cumsum(axis=0) < 0.95)[0].max()
esch.plot(S[None, :])  # chose 90th percentile.
esch.plot(U[:, :thresh].T)
# plt.show()
# %%
plt.plot(U[:, 2])

# %%
# %%
# @partial(vmap, in_axes=(None, 0))
# def scope_fn(params, x):
#     embeds = mi.model.embed_fn(params.embeds, x)
#     step_fn = partial(mi.model.block_fn, dropout=0.0)
#     z, acts = lax.scan(step_fn, embeds, params.blocks)
#     logits = mi.model.base_n_pos_weigh(z @ params.unbeds, cfg.prime)
#     return embeds, acts, logits  # reorder


# %% Hinton plots
# embeds, (attn_acts, ffwd_acts), logits = scope_fn(params, ds.x)
# mi.plots.hinton_fn(attn_acts.k)

# plot_head_activations(attn_acts.k.mean(axis=0)[0])
# mi.plots.plot_head_activations(attn_acts.wei.mean(axis=0)[0])
# mi.plots.plot_head_activations(attn_acts.wei.mean(axis=0)[1])
# %%
# mi.plots.hinton_fn(attn_acts.wei.mean(axis=0)[0].mean(axis=-1))


def wei_fn(acts):  # t: time, s: sample, l: layer, h: head
    qk = acts.q @ rearrange(acts.k, "t s l h tok c -> t s l h c tok")
    qk /= jnp.sqrt(acts.k.shape[-1])
    wei = nn.softmax(qk, axis=-1)
    return wei


#
# wei = wei_fn(acts)

# mi.model.apply_fn(params, ds.x)
# esch.plot(
# rearrange(acts.wei, "time (a b) layer head fst snd -> time a b layer head fst snd", a=cfg.prime, b=cfg.prime)[
# :, :, :, 0, 0, 0, 1  # time, a, b, layer, head, fst, snd
# ],
# animated=True,
# path="noah.svg",
# xlabel="First digit (a)",
# ylabel="Second digit (b)",
# xticks=[(0, str(0)), (cfg.prime - 1, str(cfg.prime - 1))],
# yticks=[(0, str(0)), (cfg.prime - 1, str(cfg.prime - 1))],
# rate=100,
# )

# %%
# def attention_hintons(attn_acts, layer, a, b):
# wei = attn_acts.wei
# fig, axes = plt.subplots(ncols=cfg.hyper.heads, figsize=(14, 8))
# for i, ax in enumerate(axes):  # type: ignore
# x = wei[:, layer, i, a, b].reshape(cfg.prime, cfg.prime)
# mi.plots.hinton_fn(x, ax, scale=1)
# ax.set_title(f"Head {i}")
# ax.set_ylabel("First digit")
# ax.set_xlabel("Second digit")
# add y ticks at 0 and 36 with those numbers
# add text to the right side of plot (rotated 90) giving the description hellow world
# fig.text(1, 0.5, f"Attention from digit a to b", va="center", rotation=90)
# plt.tight_layout()
# plt.savefig(f"paper/figs/attention_layer_{layer}.svg", format="svg", bbox_inches="tight")
# plt.close()


# attention_hintons(attn_acts, 0, 1, 1)
# attention_hintons(attn_acts, 0, 0, 1)
# fig, ax = plt.subplots(figsize=(8, 8))
# mi.plots.hinton_fn(attn_acts.wei[:, 0, 1, 1, 1].reshape(p, p), ax)
# ax.set_yticks([0, p - 1])
# ax.set_xticks([0, p - 1])
# plt.savefig("paper/figs/attention_one.svg", format="svg", bbox_inches="tight")


# %%
# def plot_sample_activations(embeds, attn_acts, ffwd_acts, logits, ds, i=0):
# def plot_block(acts, ax):
# mi.plots.hinton_fn(acts, ax)

# Plot 1: Embeddings
# plt.figure(figsize=(12, 4))
# plt.title("Embeddings")
# plot_block(embeds[i], plt.gca())
# plt.ylabel(ds.train.x[i])
# plt.yticks(range(len(ds.train.x[i])), ds.train.x[i])
# plt.gca().tick_params(axis="y", length=0)
# plt.tight_layout()
# plt.show()

# Plot 2: Attention Activations
# fig, axes = plt.subplots(cfg.hyper.depth, 1, figsize=(12, 4 * cfg.hyper.depth))
# fig.suptitle("Attention Activations", fontsize=16)
# for j, ax in enumerate(axes):  # type: ignore
# plot_block(attn_acts[i][j], ax)
# ax.set_title(f"Attn Layer {j+1}")
# plt.tight_layout()
# plt.show()
#
# Plot 3: Feedforward Activations
# fig, axes = plt.subplots(cfg.hyper.depth, 1, figsize=(12, 4 * cfg.hyper.depth))
# fig.suptitle("Feedforward Activations", fontsize=16)
# for j, ax in enumerate(axes):  # type: ignore
# plot_block(ffwd_acts[i][j], ax)
# ax.set_title(f"FFWD Layer {j+1}")
# plt.tight_layout()
# plt.show()
#
# Plot 4: Z
# plt.figure(figsize=(12, 4))
# plot_block(logits[i][::-1], plt.gca())  # IMPORANT: MATRIX INDEX AND HINTON PLOT have different indexing
# the x-ticks should be ds.train.y[i]
# plt.xticks(range(len(ds.info.tasks)), ds.info.tasks[:-1] + ["ℙ"])
# plt.gca().tick_params(axis='x', rotation=90)
# plt.yticks(range(len(ds.train.x[i])), ds.train.x[i][::-1])
# plt.tight_layout()
# plt.show()


# %%
# plot_sample_activations(embeds, attn_acts, ffwd_acts, logits, ds, 113)


# %%
# mi.plots.plot_run(metrics, ds, cfg)  # type: ignore
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

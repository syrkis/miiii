# %% notebook.py
#   miiii notebook
# by: Noah Syrkis


# %% Imports
from copy import deepcopy
import sys

import esch
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange
from jax import random
from oeis import oeis
from functools import partial

import miiii as mi


# %% Functions
def fourier_basis(p):  # TODO This is a bit wrong
    freqs = jnp.arange(1, p // 2 + 1)[:, None]
    phase = 2 * jnp.pi * freqs * jnp.arange(p) / p
    F = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)])
    return F / jnp.linalg.norm(F, axis=1, keepdims=True)


def plot_training(metrics, acts, y, cfg):
    plot_f1_final_tasks(metrics, cfg)
    plot_f1_tasks(metrics, cfg)
    # y_hat = (nn.sigmoid(acts.logits) > 0.5).astype(jnp.int8)[:, -1]
    # data = jnp.array([y_hat.mean(0)[None, :], y.mean(0)[None, :]])
    # esch.plot(data)


def plot_f1_final_tasks(metrics, cfg):
    # axis stuff
    left = esch.EdgeConfig(ticks=[(0, "Train"), (1, "Test")], show_on="first")
    ticks = [(i, str(task)) for i, task in enumerate(oeis["A000040"][1 : metrics.train.f1.shape[1] + 1])]
    bottom = esch.EdgeConfig(ticks=ticks, show_on="all")  # type: ignore
    edge = esch.EdgeConfigs(left=left, bottom=bottom)
    data = jnp.concat((metrics.train.f1[-1][None, :], metrics.valid.f1[-1][None, :]), axis=0)
    esch.plot(data, edge=edge, path=f"paper/figs/miiii_f1_tasks_{cfg.p}_{cfg.epochs}_last.svg")


def plot_f1_tasks(metrics, cfg):
    # axis stuff
    left = esch.EdgeConfig(ticks=[(0, "2"), (28, "109")], show_on="first", label="Task")
    bottom = esch.EdgeConfig(label="Time", show_on="all", ticks=[(0, str(0)), (99, f"{cfg.epochs - 1:,}")])
    edge = esch.EdgeConfigs(left=left, top=bottom)

    data = jnp.clip(metrics.train.loss.T[:, :: (cfg.epochs // 100)], 0, 0.1)
    esch.plot(data, path=f"paper/figs/miiii_f1_tasks_{cfg.p}_{cfg.epochs}.svg", edge=edge)


def plot_attention_samples(acts, slice, cfg):
    # axis stuff
    top = esch.EdgeConfig(label=[f"Head {i}" for i in range(4)], show_on="all")
    left = esch.EdgeConfig(label="𝑥₁", show_on="first")
    bottom = esch.EdgeConfig(label="𝑥₀", show_on="all")
    edge = esch.EdgeConfigs(top=top, left=left, bottom=bottom)

    reshape = "(fst snd) layer head a b -> head fst snd layer a b"
    wei = rearrange(acts.wei, reshape, fst=cfg.p, snd=cfg.p)[..., -1, -1, 0]
    esch.plot(wei[:, :slice, :slice], edge=edge, path=f"paper/figs/weis_{cfg.project}_{cfg.p}_slice_{slice}.svg")


def plot_n_neurons(acts, slice, n, cfg):
    reshape = "(a b) neuron -> neuron a b"
    neurons = rearrange(acts.ffwd.squeeze()[:, -1], reshape, a=cfg.p, b=cfg.p)[:n][:, :slice, :slice]
    esch.plot(neurons, path=f"paper/figs/ffwd_{cfg.project}_{cfg.p}_{n}_neurons_slice_{slice}.svg")


def plot_x(x, cfg):
    left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(11)], show_on="first")
    top = esch.EdgeConfig(label=[f"{i}" for i in range(11)], show_on="all")
    edge = esch.EdgeConfigs(left=left, bottom=top)
    data = rearrange(x[:, :-1], "(a b) c -> b a c", a=cfg.p, b=cfg.p)[:11, :11, :11]
    esch.plot(data, edge=edge, path=f"paper/figs/ds_{cfg.project}_{cfg.p}_x.svg")


def plot_y(y, cfg):
    # %% Plotting y
    top = esch.EdgeConfig(label=[f"Task {p}" for p in [2, 3, 5, 7]] + ["Modular addition task"], show_on="all")
    bottom = esch.EdgeConfig(label="𝑥₀", show_on="all")
    left = esch.EdgeConfig(label="𝑥₁", show_on="all")
    # right = esch.EdgeConfig(label="", show_on="last")
    edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
    data = rearrange(y, "(a b) t -> t a b", a=cfg.p, b=cfg.p)[:4, :11, :11]
    # nanda_y = (jnp.arange(121) // 11).reshape(11, 11)
    nanda_cfg = deepcopy(cfg)
    nanda_cfg.p = 11
    nanda_cfg.project = "nanda"
    nanda_ds = mi.tasks.task_fn(random.PRNGKey(0), nanda_cfg)
    nanda_y = jnp.concat((nanda_ds.y_train, nanda_ds.y_valid))[nanda_ds.idxs.argsort()].reshape(1, 11, 11) / 10
    data = jnp.concatenate((data, nanda_y), axis=0)
    esch.plot(data, edge=edge, path="paper/figs/ds_11_y.svg")


# %% polar plots
def polar_plot():
    primes = jnp.array(oeis["A000040"][1:1000])
    ps = jnp.array(primes[primes < (113**2)])
    _11s = jnp.arange(0, 113**2, 11)
    _7_23 = jnp.concat((jnp.arange(0, 113**2, 7), jnp.arange(0, 113**2, 23)))
    plt.style.use("default")
    mi.plots.small_multiples(fnames=["n", "t", "n"], seqs=[_7_23, _11s, ps], f_name="polar", n_rows=1, n_cols=3)
    # remove plot
    plt.close()


# %%
def embedding_report_fn(state, cfg):
    W_E = state.params.embeds.tok_emb[:-1]  # p x latent_dim
    U, S, V = jnp.linalg.svd(W_E)  # p x p, p, latent_dim x latent_dim
    F = fourier_basis(cfg.p)
    singular_values(S, cfg)
    top_singular_vectors(U, S, cfg)
    embed_in_fourier(W_E, F, S, cfg)


def embed_in_fourier(W_E, F, S, cfg):
    tmp = F @ W_E
    fifty = jnp.where((S / S.sum()).cumsum() < 0.5)[0].max()
    esch.plot(tmp if tmp.shape[0] < tmp.shape[1] else tmp.T)
    most_significat = jnp.linalg.norm(F @ W_E, axis=-1).argsort()[-fifty:]
    # print(most_significat)
    ticks = [(i.item(), "cos " + str(i // 2)) for i in most_significat if i % 2 == 1]
    bottom = esch.EdgeConfig(ticks=ticks, show_on="first")
    # left = esch.EdgeConfig(ticks=[(0, "constant")], show_on="first")
    ticks = [(i.item(), "sin " + str(i // 2)) for i in most_significat if i % 2 == 0] + [(0, "constant")]
    top = esch.EdgeConfig(ticks=ticks, show_on="all")
    edge = esch.EdgeConfigs(bottom=bottom, top=top)
    esch.plot(
        (jnp.linalg.norm(F @ W_E, axis=-1))[None, :], edge=edge, path=f"paper/figs/{cfg.project}_{cfg.p}_F_W_E.svg"
    )

    key_embed = (F @ W_E)[most_significat]
    esch.plot(key_embed @ key_embed.T, path=f"paper/figs/{cfg.project}_{cfg.p}_key_embed.svg")


def singular_values(S, cfg):
    fifty = jnp.where((S / S.sum()).cumsum() < 0.5)[0].max()
    ninety = jnp.where((S / S.sum()).cumsum() < 0.9)[0].max()
    bottom = esch.EdgeConfig(ticks=[(int(fifty.item()), "0.5"), (ninety.item(), "0.9")], show_on="first")
    left = esch.EdgeConfig(ticks=[(0, "S")], show_on="first")
    edge = esch.EdgeConfigs(bottom=bottom, left=left)
    esch.plot(S[None, :], path=f"paper/figs/{cfg.project}_{cfg.p}_S_top_37.svg", edge=edge)


def top_singular_vectors(U, S, cfg):
    def subscript(i):
        return chr(0x2080 + i)

    fifty = jnp.where((S / S.sum()).cumsum() < 0.5)[0].max()
    left = esch.EdgeConfig(ticks=[(i, "𝘶" + subscript(i)) for i in range(fifty)], show_on="first")
    edge = esch.EdgeConfigs(left=left)
    esch.plot(U[:, :fifty].T, path=f"paper/figs/{cfg.project}_{cfg.p}_U_top_{fifty}.svg", edge=edge)


def data_report_fn(x, y, cfg):
    plot_x(x, cfg)
    plot_y(y, cfg)
    polar_plot()


def report_fn(hash, slice):
    # %% Setup
    state, (metrics, _) = mi.utils.get_metrics_and_params(hash)  # get a run
    cfg = mi.utils.construct_cfg_from_hash(hash)  # and associated config
    ds = mi.tasks.task_fn(random.PRNGKey(0), cfg)  # and the dataset
    merge = lambda x, y: jnp.concat((x, y), axis=0)[ds.idxs.argsort()]  # noqa
    x, y = map(merge, (ds.x_train, ds.x_valid), (ds.y_train, ds.y_valid))

    data_report_fn(x, y, cfg) if cfg.project == "miiii" else None
    embedding_report_fn(state, cfg)

    apply = partial(mi.model.apply_fn(cfg, ds, eval=True), random.PRNGKey(0))
    acts = apply(state.params, x)  # and finally the activations

    # %% Create arrays
    # W_neur = W_E @ state.params.attn.v[0] @ state.params.attn.o[0] @ state.params.ffwd.w_in  # noqa
    # W_logit = state.params.ffwd.w_out[0] @ state.params.unbeds  # noqa
    # U, S, V = jnp.linalg.svd(W_E)
    # F = fourier_basis(cfg.p)

    # %% Plot arrays
    plot_training(metrics, acts, y, cfg) if cfg.project == "miiii" else None
    plot_attention_samples(acts, slice, cfg)
    plot_n_neurons(acts, slice, 5, cfg)
    # esch.plot((state.params.ffwd.w_out @ state.params.unbeds).squeeze().T)


# %%
if __name__ == "__main__" and "ipykernel" in sys.argv[0]:
    hash = "08aea9ab146c4c22917a1920"
    slice = 23
    report_fn(hash, slice)

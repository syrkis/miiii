# %% paper.py
#   generating plots for the paper (and some exploratory stuff)
# by: Noah Syrkis


# %% Imports ######################################################################
import miiii as mi

import esch
import jax.numpy as jnp
import matplotlib.pyplot as plt

from oeis import oeis
from einops import rearrange
from functools import partial
from jax import random, tree, vmap
from jax.numpy import fft
from collections import Counter


# %% Utils ####################################################################
def load_hash(hash, task="miiii"):
    state, metrics, scope, cfg = mi.utils.get_metrics_and_params(hash)
    ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
    apply = partial(mi.model.apply_fn(cfg, ds, task, False), random.PRNGKey(0))
    x = jnp.concat((ds.x.train, ds.x.eval))[ds.idxs.argsort()]
    acts = apply(state.params, x)
    return state, metrics, scope, cfg, ds, task, apply, x, acts


def run_fn(hashes):
    # plot svd
    # emb_svd(state.params, cfg, f_hash)

    # plot training "curves"
    # mi.plots.plot_run(metrics, ds, cfg, task, f_hash, font_size=16)  # plot training "curves"

    # FFT plot
    # emb_fft(state.params, cfg, f_hash)

    # Plot active freqs through time
    # omega_fn(cfg, (jnp.abs(scope.neuron_freqs)))
    pass

    # plot first five neurons


# %% Constants ####################################################################
rng = random.PRNGKey(0)
slice = 37
m_hash = "d4bfd7f829ed4a398f3b0a54"  # hash of masked miiii
f_hash = "5cc8559431664714b27fab75"  # hash of miiii task
s_hash = "7c2a10494ff64e66a9af2731"  # shuffled miiii task  10k epochs currently
p_hash = "0c848c1444264cbfa1a4de6e"  # hash of nanda task
data = {hash: load_hash(hash) for hash in [f_hash, m_hash, s_hash]}


# %% Functions ##################################################################
def emb_svd(params, cfg, hash):
    tok_emb = params.embeds.tok_emb[: cfg.p]
    U, S, V = jnp.linalg.svd(tok_emb)
    quantiles = (S / S.sum()).cumsum()
    esch.plot(U.T[quantiles < 0.5], path=f"paper/figs/{hash}U.svg")
    esch.plot(S[None, :])


def emb_fft(params, cfg, hash):
    m, f, s = fft_fn(params.embeds.tok_emb[:-1])
    esch.plot(m)
    esch.plot(fft_fn(mi.model.initializer(rng, params.embeds.tok_emb[:-1].shape))[0])


def fft_fn(matrix):
    # Compute 2D FFT
    fft_2d = fft.rfft2(matrix.T).T
    magnitude_spectrum = jnp.abs(fft_2d)
    # Center everything
    magnitude_spectrum_centered = fft.fftshift(magnitude_spectrum)
    # frequency activations
    freq_activations = jnp.linalg.norm(magnitude_spectrum_centered, axis=1)
    significant_freqs = freq_activations > freq_activations.mean()
    return magnitude_spectrum_centered, freq_activations, significant_freqs


def omega_aux(freqs, length=150, kernel_size=3):
    epochs = freqs.shape[0]
    # kernel_size = epochs // length
    conv = lambda row: jnp.convolve(row, jnp.ones(kernel_size) / kernel_size, mode="valid")  # noqa
    freq_series = vmap(conv)(jnp.abs(freqs).T)  # smooth this stuff
    freq_series = freq_series[1:, :: epochs // length]
    freq_series /= freq_series.sum(axis=1, keepdims=True)

    freq_variance = freq_series.var(axis=0)
    freq_active = (freq_series > (freq_series.mean() + freq_series.std())).sum(0)  # noqa
    # print(freq_active)

    # return the line as well
    return freq_series, freq_variance, freq_active


def omega_series_fn(freqs, label_top, label_bottom, fname="finding.svg"):
    # neuron_freqs = omega_aux(neuron_freqs)

    right = esch.EdgeConfig(label="Time", show_on="all")
    left = esch.EdgeConfig(label="Frequency" if label_top != "" else "", show_on="all")
    top = esch.EdgeConfig(label=label_top, show_on="first")
    bottom = esch.EdgeConfig(label=label_bottom, show_on="last")
    edge = esch.EdgeConfigs(left=left, top=top, bottom=bottom, right=right)
    data = freqs**2
    esch.plot(data / data.max(1)[:, None], path=f"paper/{fname}.svg", edge=edge, font_size=24)

    # left = esch.EdgeConfig(label=["ω σ²", "|ω > μ + 2σ|"], show_on="all")
    # edge = esch.EdgeConfigs(left=left)

    # tmp = neuron_freqs.var(axis=0)
    # data = neuron_freqs
    # v1 = tmp[None, :]
    # v3 = (data > (data.mean() + 2 * data.std()))[1:].astype(jnp.float16).sum(0)[None, :]
    # v = jnp.concat((v1, v3))
    # esch.plot(v / v.mean(1, keepdims=True), edge=edge, path="paper/finding2.svg", font_size=24)
    # print((v3 == 0).sum())
    # print(Counter(v3.tolist()))
    # v.shape


# %% work space #################################################################
f_data, m_data, s_data = data.values()
f_scope, m_scope, s_scope = f_data[2], m_data[2], s_data[2]
f_acts, m_acts, s_acts = f_data[-1], m_data[-1], s_data[-1]

# %%
esch.plot((f_scope.neuron_freqs / f_scope.neuron_freqs.max(1, keepdims=True))[::4].T)
f_scope.neuron_freqs

# %%
# f_acts.ffwd.squeeze()[:, -1].shape
neurs = jnp.abs(fft.fft2(rearrange(f_acts.ffwd.squeeze()[:, -1], "(x0 x1) h -> h x0 x1", x0=113, x1=113)))[..., 1:, 1:]
esch.plot(((neurs / neurs.max()) > 0.5).sum((0, 1))[None, :])


f_neurs = jnp.abs(fft.fft2(rearrange(f_acts.ffwd.squeeze()[:, -1], "(x0 x1) h -> h x0 x1", x0=113, x1=113)))[
    ..., 1:, 1:
]
esch.plot(f_neurs.mean((0, 2))[None, :])


# %%
esch.plot(
    jnp.abs(fft.rfft2(rearrange(f_acts.ffwd.squeeze()[:, -1], "(x0 x1) h -> h x0 x1", x0=113, x1=113))[:3])[
        ..., 1 : 113 // 2, 1 : 113 // 2
    ],
)
# %% Omega plots
for hash in [f_hash, s_hash]:  # ,s_hash]:
    state, metrics, scope, cfg, ds, task, apply, x, acts = data[hash]
    neurons = rearrange(acts.ffwd.squeeze()[:, -1], "(x0 x1) h ->h x0 x1", x0=cfg.p, x1=cfg.p)
    esch.plot(neurons[:3, :slice, :slice], path=f"paper/{hash}_tmp_1.svg")
    esch.plot(
        jnp.abs(fft.rfft2(rearrange(acts.ffwd.squeeze()[:, -1], "(x0 x1) h -> h x0 x1", x0=113, x1=113))[:3])[
            ..., 1 : 113 // 2, 1 : 113 // 2
        ],
        path=f"paper/{hash}_astrid.svg",
    )

# %%
# esch.plot(omega_aux(f_scope.neuron_freqs)[0])
f_neurs = f_scope.neuron_freqs
esch.plot((f_neurs / f_neurs.max(0))[::4].T)


# %%


# %%
f_neurs = f_neurs / f_neurs.max(axis=(0, 2), keepdims=True)
esch.plot((f_neurs > 0.5).sum((0, 1))[None, :])
esch.plot(f_neurs[0])

# %%

# left = esch.EdgeConfig(ticks=[(0, "𝑎")], show_on="all")
# right = esch.EdgeConfig(ticks=[(1, "𝑏")], show_on="all")
# edge = esch.EdgeConfigs(left=left, right=right)
# esch.plot(freq_active[0][None, :], font_size=30, path="paper/figs/omega.svg")
# omega_fn(cfg, scope.neuron_freqs)
# plt.plot(freq_active.T)

# freq_series = jnp.stack(
# (omega_aux(f_scope.neuron_freqs)[0], omega_aux(s_scope.neuron_freqs)[0], omega_aux(m_scope.neuron_freqs)[0])
# )
# omega_series_fn(freq_series[0], "Time", "", fname="omega-series-1")
# omega_series_fn(freq_series[1], "", "", fname="omega-series-2")
# omega_series_fn(freq_series[2], "", "", fname="omega-series-3")

# %%
# %% Training curves
for hash in [f_hash, m_hash, s_hash]:
    state, metrics, scope, cfg, ds, task, apply, x, acts = data[hash]
    mi.plots.plot_run(metrics, ds, cfg, task, hash, font_size=16, log_axis=False)
# %% nanda plots ###########################################################

# %% p task
# p_hash = "0c848c1444264cbfa1a4de6e"
# p_state, p_metrics, p_cfg = mi.utils.get_metrics_and_params(p_hash, task_span="prime")
# p_ds, p_task = mi.tasks.task_fn(rng, p_cfg, "remainder", "prime")
# p_apply = partial(mi.model.apply_fn(p_cfg, p_ds, p_task, False), random.PRNGKey(0))
# p_x = jnp.concat((p_ds.x.train, p_ds.x.eval))[p_ds.idxs.argsort()]
# p_acts = p_apply(p_state.params, p_x)


# %% Positional embeddings analysis
# f_pos_emb = f_state.params.embeds.pos_emb[:2][:, :slice]
# p_pos_emb = p_state.params.embeds.pos_emb[:2][:, :slice]
# pos_emb = jnp.stack((p_pos_emb, f_pos_emb), axis=0)
label = f"First {slice} dimensions of position embeddings for the factors (top) and prime (bottom) tasks"
left = esch.EdgeConfig(label=["nanda", "miiii"], show_on="all")
edge = esch.EdgeConfigs(left=left)
esch.plot(pos_emb, edge=edge, path="paper/figs/pos_emb.svg")


# %% Token embedding exploratoray analysis
f_tok_emb = f_state.params.embeds.tok_emb[: f_cfg.p]
f_U, f_S, f_V = jnp.linalg.svd(f_tok_emb)
# p_tok_emb = p_state.params.embeds.tok_emb[: p_cfg.p]
# p_U, p_S, p_V = jnp.linalg.svd(p_tok_emb)

f_S_50 = jnp.where((f_S / f_S.sum()).cumsum() < 0.5)[0].max()
# p_S_50 = jnp.where((p_S / p_S.sum()).cumsum() < 0.5)[0].max()
f_S_90 = jnp.where((f_S / f_S.sum()).cumsum() < 0.9)[0].max()
# p_S_90 = jnp.where((p_S / p_S.sum()).cumsum() < 0.9)[0].max()
# S = jnp.stack((p_S / p_S.sum(), f_S / f_S.sum()), axis=0).reshape((2, 1, -1))[:, :, :83]

top = esch.EdgeConfig(ticks=[(f_S_50.item(), "0.5"), (f_S_90.item(), "0.9")], show_on="first")
left = esch.EdgeConfig(label=["nanda", "miiii"], show_on="all")
# bottom = esch.EdgeConfig(ticks=[(p_S_50.item(), "0.5"), (p_S_90.item(), "0.9")], show_on="last")
# edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
# esch.plot(S, edge=edge, path="paper/figs/S.svg")


esch.plot(f_U[:, : f_S_50.item()].T, path="paper/figs/f_U.svg")
# esch.plot(p_U[:, : p_S_50.item()].T, path="paper/figs/p_U.svg")
# %% plots
mi.plots.plot_run(f_metrics, f_ds, f_cfg, f_task, f_hash, font_size=16)


# %% Embeddings fourier analsysis
# p_m, p_f, p_s = mi.plots.fourier_analysis(p_state.params.embeds.tok_emb[:-1])
f_m, f_f, f_s = mi.plots.fourier_analysis(f_state.params.embeds.tok_emb[:-1])
r_m, r_f, r_s = mi.plots.fourier_analysis(mi.model.initializer(rng, f_state.params.embeds.tok_emb[:-1].shape))


# %%
def fourier_plots(m, f, s, name):
    esch.plot(m, path=f"paper/figs/fourier_{name}_m.svg")
    ticks_bottom = [(i.item(), f"cos {i//2}") for i in jnp.where(s)[0] if i % 2 == 1]
    ticks_top = [(0, "const")] + [(i.item(), f"sin {i//2}") for i in jnp.where(s)[0] if i % 2 == 0]
    top = esch.EdgeConfig(ticks=ticks_top, show_on="all")  # type: ignore
    bottom = esch.EdgeConfig(ticks=ticks_bottom, show_on="all")
    edge = esch.EdgeConfigs(top=top, bottom=bottom)
    if name != "r":
        esch.plot(f[None, :], path=f"paper/figs/fourier_{name}_f.svg", edge=edge, font_size=8)
    else:
        esch.plot(f[None, :], path=f"paper/figs/fourier_{name}_f.svg")


# fourier_plots(p_m, p_f, p_s, "p")
fourier_plots(f_m, f_f, f_s, "f")
fourier_plots(r_m, r_f, r_s, "r")

# %%
# fourier_basis = fft.rfft2(jnp.eye(p_cfg.p))
# esch.plot((fourier_basis.T @ fourier_basis.conj()).imag)
# fourier_basis

# %%
heads = rearrange(f_acts.wei, "(a b) l h x0 x1 -> h a b l x0 x1", a=f_cfg.p, b=f_cfg.p).squeeze()[
    :, :slice, :slice, -1, 0
]
esch.plot(heads[0])
# p_acts.wei.shape


# %%
neurons = rearrange(f_acts.ffwd, "(x0 x1) l p n -> n x0 x1 l p", x0=f_cfg.p, x1=f_cfg.p).squeeze()[..., -1]
top = esch.EdgeConfig(ticks=[(0, "0"), (36, "36")], show_on="all", label="𝑥₀")
left = esch.EdgeConfig(ticks=[(0, "0"), (36, "36")], show_on="all", label="𝑥₁")
edge = esch.EdgeConfigs(top=top, left=left)
esch.plot(heads[0], edge=edge, path="paper/figs/plot_intro_miiii.svg")

# %%
# print([round(acc, 2) for acc in f_metrics.valid.acc[-1].round(2).tolist()])


# esch.plot(p_state.params.embeds.tok_emb, path="paper/figs/tok_emb_prime.svg")
# esch.plot(p_state.params.embeds.pos_emb, path="paper/figs/pos_emb_prime.svg")
# %%
# esch.plot(p_state.params.attn.v.squeeze(), path="paper/figs/attn_v_prime.svg")
# esch.plot(p_state.params.attn.k.squeeze(), path="paper/figs/attn_k_prime.svg")
# esch.plot(p_state.params.attn.q.squeeze(), path="paper/figs/attn_q_prime.svg")


# %%
# esch.plot(p_state.params.ffwd.w_in.squeeze().T, path="paper/figs/ffwd_w_in_prime.svg")
# esch.plot(p_state.params.ffwd.w_out.squeeze(), path="paper/figs/ffwd_w_out_prime.svg")

# %%
# esch.plot(p_state.params.unbeds, path="paper/figs/unbeds_prime.svg")


# %%
#
leafs, struct = tree.flatten(f_scope.grad_norms)
ticks = [(i, w) for i, w in enumerate("emb_pos emb_tok k o q v w_in w_out unbeds".split())]
right = esch.EdgeConfig(ticks=ticks, show_on="all")  # type: ignore
top = esch.EdgeConfig(ticks=[(0, "1"), (49, str(f_cfg.epochs))], show_on="all", label="Time (linear)")
left = esch.EdgeConfig(label="Gradient Norm (L2)", show_on="all")
edge = esch.EdgeConfigs(right=right, left=left, top=top)
data = jnp.array(leafs)[:, 1000 :: f_cfg.epochs // 50]
data = data / data.max(axis=1, keepdims=True)
# data = data / data.sum(axis=0, keepdims=True)
data = data[[4, 5, 0, 1, 2, 3, 6, 7, 8], :]
esch.plot(data, edge=edge, path="paper/figs/grads_norms_miiii.svg")
# struct

# %%
dft = jnp.fft.fft(jnp.eye(f_cfg.p))
esch.plot(dft.real, path="paper/figs/real_dft.svg")
# dft = dft / jnp.linalg.norm(dft, axis=1)
esch.plot(dft @ f_state.params.embeds.tok_emb[:-1])


# %%
esch.plot(jnp.abs(dft[f_s.repeat(2)[1:]][::2].mean(0))[None, 1:])


# %%  W_NEUR
f_neurs = rearrange(f_acts.ffwd.squeeze()[:, -1], "(x0 x1) n -> n x0 x1", x0=f_cfg.p, x1=f_cfg.p)
esch.plot(dft @ f_neurs[0] @ dft.T)

# %% Neuron frac explained by Freq
neuron_freq_norm = jnp.zeros((f_cfg.p // 2, f_cfg.latent_dim * 4))
for freq in range(0, f_cfg.p // 2):
    for x in [0, 2 * freq - 1, 2 * freq]:
        for y in [0, 2 * freq - 1, 2 * freq]:
            tmp = neuron_freq_norm[freq] + f_neurs[:, x, y] ** 2
            neuron_freq_norm.at[freq].set(tmp)
neuron_freq_norm = neuron_freq_norm / (f_neurs**2).sum(axis=(-1, -2), keepdims=True)

# %%
# jnp.unique(neuron_freq_norm)
#
# esch.plot((dft @ f_neurs @ dft.T)[1])

esch.plot(jnp.abs(fft.fft2(f_neurs).max((0, 1)))[None, 1:])  # THIS IS INTERSTING
esch.plot(jnp.abs(fft.fft2(f_neurs).mean((0, 1)))[None, 1:])  # THIS IS INTERSTING
# f_neurs.shape, fft.rfft2(f_neurs).shape
# %%
data = jnp.abs(dft @ f_neurs).mean(2)

esch.plot(data)
data.shape
# %%
# data = data - data.mean(axis=0)
esch.plot(data.sum(1)[None, :])


# %%
# esch.plot(data.mean(0)[None, :100].max(1).sort(descending=True))
frac_explainedby_top = (data.sort(0)[-4:, :].sum(0) / (data.sum(0) + 1e-8)).sort()[None, :]
esch.plot(frac_explainedby_top[None, :])


# %% similar analysis on output logits.


# %% Progress measures
# rearrange(p_acts.logits, "(a b) c -> a b c", a=p_cfg.p, b=p_cfg.p).shape

# %%
# f_acts.ffwd.squeeze().shape, f_neurs.shape


# %%
data = jnp.abs(scope.neuron_freqs).T
# data = data**abs
length = 150
window_size = (cfg.epochs // length) // 2
data = jnp.array([jnp.convolve(row, jnp.ones(window_size) / window_size, mode="valid") for row in data])
data = data / data.sum(axis=1, keepdims=True)
tmp = data.var(axis=0)
# Simple moving average
kernel = jnp.ones(window_size) / window_size
tmp = jnp.convolve(tmp, kernel, mode="valid")
tmp = tmp / tmp.max()

left = esch.EdgeConfig(label="Frequency", show_on="all")
top = esch.EdgeConfig(label="Time", show_on="last")
edge = esch.EdgeConfigs(left=left, top=top)
esch.plot(data[1:, :: cfg.epochs // length] ** 1.5, path="paper/finding.svg", edge=edge, font_size=24)
left = esch.EdgeConfig(label=["ω σ²", "|ω > μ + 2σ|"], show_on="all")
edge = esch.EdgeConfigs(left=left)


v1 = tmp[None, :: cfg.epochs // length]
# v2 = (data > data.mean() + data.std())[1:, :: f_cfg.epochs // length].astype(jnp.float16).sum(0)[None, :]
v3 = (data > data.mean() + 2 * data.std())[1:, :: cfg.epochs // length].astype(jnp.float16).sum(0)[None, :]
v = jnp.stack((v1, v3))
esch.plot(v / v.max(2, keepdims=True), edge=edge, path="paper/finding2.svg", font_size=24)
v.shape


# %%
# v3 in jnp.unique(v3).sort()[-3:]


# %%
#
# fig, ax = plt.subplots(figsize=(12, 2), dpi=100)
# ax.plot(tmp, c="w")
# # x in log scale
# # ax.set_xscale("log")
# # crop first 1000 x values
# ax.set_xlim(100, None)
# # hide x ticks and labels
# ax.set_xticks([])
# # y ticks and labels should only be 0 and 1
# ax.set_yticks([0, 1])
# ax.set_yticklabels(["0", "1"])

# %%


# %% paper.py
#   generates all plots for paper
# by: Noah Syrkis

# %% Imports

"""
# %% Constants
factors_hash = "713e658dbfca4ab98c6e53ed"
# prime_hash = "713e658dbfca4ab98c6e53ed"


# Diagrams and illustrations
# %% X plots
#
cfg = mi.utils.Conf(p=11)
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
x = jnp.concat((ds.x.train, ds.x.eval), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y.train, ds.y.eval), axis=0)[ds.idxs.argsort()]
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
bottom = esch.EdgeConfig(label=[str(i) for i in range(cfg.p)], show_on="all")
edge = esch.EdgeConfigs(left=left, bottom=bottom)
esch.plot(
    rearrange(x[:, :2], "(x1 x0) seq ->  x0 x1 seq ", x0=cfg.p, x1=cfg.p),
    edge=edge,
    path="figs/x_11_plot.svg",
    font_size=10,
)

# %% Y plots
nanda_cfg = mi.utils.Conf(p=11)
nanda_ds, _ = mi.tasks.task_fn(random.PRNGKey(0), nanda_cfg, "remainder", "prime")
nanda_y = jnp.concat((nanda_ds.y.train, nanda_ds.y.eval), axis=0)[nanda_ds.idxs.argsort()].reshape(
    (nanda_cfg.p, nanda_cfg.p)
)
primes = jnp.array(oeis["A000040"][1 : y.shape[1] + 1])
bottom = esch.EdgeConfig(label=[f"{factor} remainder" for factor in primes] + ["𝑝 remainder"], show_on="all")
top = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
edge = esch.EdgeConfigs(top=top, left=left, bottom=bottom)
data = jnp.concat((rearrange(y, "(x0 x1) task ->  task x0 x1 ", x0=cfg.p, x1=cfg.p), nanda_y[None, ...]), axis=0)
# data /= data.max(axis=(1, 2))[:, None, None]
esch.plot(data, edge=edge, path="figs/y_11_plot.svg", font_size=10)


# %%


# %% Polar Plots
primes = jnp.array(oeis["A000040"][1:1000])
ps = jnp.array(primes[primes < (113**2)])
_11s = jnp.arange(0, 113**2, 11)
_7_23 = jnp.concat((jnp.arange(0, 113**2, 13), jnp.arange(0, 113**2, 23)))
plt.style.use("default")
mi.plots.small_multiples(fnames=["n", "t", "n"], seqs=[_7_23, _11s, ps], f_name="polar", n_rows=1, n_cols=3)
# remove plot
# plt.close()


# %% Constants
hash = "8817b4ba85254125b7af28ec"
slice = 43
state, metrics, cfg = mi.utils.get_metrics_and_params(hash, "factors")  # get a run
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
x = jnp.concat((ds.x.train, ds.x.eval), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y.train, ds.y.eval), axis=0)[ds.idxs.argsort()]


# %% ATTENTION WEIGHTS
apply = mi.model.apply_fn(cfg, ds, task, eval=True)
acts = apply(rng, state.params, x)
esch.plot(
    rearrange(acts.wei.squeeze(), "(x0 x1) heads from to ->  heads x0 x1 from to", x0=cfg.p, x1=cfg.p)[:, :, :, -1, 1][
        :, :slice, :slice
    ],
    path="noah.svg",
)

# %% EMBEDDINGS
U, S, V = jnp.linalg.svd(state.params.embeds.tok_emb[: cfg.p])
s = (S / S.sum()).cumsum()
_5 = jnp.where(s < 0.5)[0].max()
_9 = jnp.where(s <= 0.9)[0].max()
_99 = jnp.where(s <= 0.99)[0].max()

# %% singular values
bottom = esch.EdgeConfig(
    ticks=[(_5.item(), "0.5"), (_9.item() - 1, "0.9"), (_99.item() - 1, "0.99")],
    show_on="first",
)
left = esch.EdgeConfig(ticks=[(0, "S")], show_on="first")
edge = esch.EdgeConfigs(bottom=bottom, left=left)
esch.plot(S[None, :], edge=edge)

# %% singular vectors
esch.plot(U[:, :_5].T)


W_E = state.params.embeds.tok_emb[: cfg.p]
F = mi.utils.fourier_basis(cfg.p)
esch.plot((F @ W_E))
esch.plot(jnp.linalg.norm(F @ W_E, axis=1)[None, :])

# %% FFWD WEIGHTS
n_neurons = 6
esch.plot(
    rearrange(acts.ffwd.squeeze()[:, -1, ...], "(x0 x1) h -> h x0 x1", x0=cfg.p, x1=cfg.p)[:n_neurons][
        :, :slice, :slice
    ],
    path=f"figs/ffwd_{slice}_{n_neurons}.svg",
)


# %% Training curves
left = esch.EdgeConfig(label="Task", show_on="first")
right = esch.EdgeConfig(ticks=[(i, str(prime.item())) for i, prime in enumerate(task.primes)], show_on="first")  # type: ignore
top = esch.EdgeConfig(ticks=[(0, "1"), (100, f"{cfg.epochs:g}")], show_on="first", label="Time")
edge = esch.EdgeConfigs(right=right, top=top, left=left)
esch.plot(metrics.train.acc[: cfg.epochs // 2 : cfg.epochs // 100 // 2].T, edge=edge, path="figs/train_acc.svg")
esch.plot(metrics.valid.acc[: cfg.epochs // 2 : cfg.epochs // 100 // 2].T, edge=edge, path="figs/valid_acc.svg")

# %%
# U, S, V = jnp.linalg.svd(state.params.embeds.pos_emb)
# U.round(1)
tmp = state.params.embeds.pos_emb.round(3)

"""

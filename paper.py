# %% paper.py
#   generating plots for the paper (and some exploratory stuff)
# by: Noah Syrkis


# %% Imports ######################################################################
import miiii as mi

import esch
import jax.numpy as jnp
import matplotlib.pyplot as plt

import os
from oeis import oeis
from einops import rearrange
from functools import partial
from jax import random, tree, vmap
from jax.numpy import fft


# %% Utils ####################################################################
def load_hash(hash, task):
    # Create the directory if it doesn't exist
    task2span = {"miiii": "factors", "nanda": "prime"}
    span = task2span.get(task, "factors")
    output_dir = f"paper/figs/{hash}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    state, metrics, scope, cfg = mi.utils.get_metrics_and_params(hash, span)
    print(span)
    ds, task = mi.tasks.task_fn(rng, cfg, "remainder", span)
    apply = partial(mi.model.apply_fn(cfg, ds, task, False), random.PRNGKey(0))
    x = jnp.concat((ds.x.train, ds.x.eval))[ds.idxs.argsort()]
    acts = apply(state.params, x)
    return state, metrics, scope, cfg, ds, task, apply, x, acts


# %% Constants ####################################################################
rng = random.PRNGKey(0)
slice = 37
miiii_hash = "50115caac50c4fbfa6bce4cc"  # hash of miiii task
masks_hash = "ba88bfb237924d5091006372"  # "d4bfd7f829ed4a398f3b0a54"  # hash of masked miiii
basis_hash = "7c2a10494ff64e66a9af2731"  # basisi with shuffled y
nodro_hash = "c7f717cb50ac4762bd866831"  # hash of miiii without dropout
nanda_hash = "0c848c1444264cbfa1a4de6e"  # hash of nanda task
data = {hash: load_hash(hash, "miiii") for hash in [miiii_hash, masks_hash, basis_hash, nodro_hash]}
data[nanda_hash] = load_hash(nanda_hash, "nanda")


# %% Functions ##################################################################
def emb_svd(params, cfg, task):
    tok_emb = params.embeds.tok_emb[: cfg.p]
    U, S, V = jnp.linalg.svd(tok_emb)
    S_50 = jnp.where((S / S.sum()).cumsum() < 0.5)[0].max()
    S_90 = jnp.where((S / S.sum()).cumsum() < 0.9)[0].max()
    # S = jnp.stack((p_S / p_S.sum(), f_S / f_S.sum()), axis=0).reshape((2, 1, -1))[:, :, :83]
    quantiles = (S / S.sum()).cumsum()

    left = esch.EdgeConfig(label="Vectors", show_on="all")
    top = esch.EdgeConfig(
        label=f"Left side singular value vectors capturing 50 % of the variance ({task})", show_on="all"
    )
    edge = esch.EdgeConfigs(left=left, top=top)
    esch.mesh(U.T[quantiles < 0.5], path=f"paper/figs/{task}_U.svg", font_size=22, edge=edge)

    # mesh singular value vectors
    title = "Sorted singular values" if task == "nanda" else ""
    bottom = esch.EdgeConfig(ticks=[(S_50.item(), "0.5"), (S_90.item(), "0.9")], show_on="first")
    top = esch.EdgeConfig(label=title, show_on=[0])
    left = esch.EdgeConfig(label=task, show_on="all")
    edge = esch.EdgeConfigs(left=left, top=top, bottom=bottom)
    esch.mesh(S[None, :], edge=edge, path=f"paper/figs/{task}_S.svg", font_size=24)


def emb_fft(params, cfg, hash):
    m, f, s = fft_fn(params.embeds.tok_emb[:-1])
    esch.mesh(m)
    esch.mesh(fft_fn(mi.model.initializer(rng, params.embeds.tok_emb[:-1].shape))[0])


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


def plot_neurs(neurs, cfg, task):
    neurs = rearrange(neurs[:, 0, -1, ...], "(x0 x1) n -> n x0 x1", x0=cfg.p, x1=cfg.p)
    left = esch.EdgeConfig(label="𝑥₀", show_on="first")
    bottom = esch.EdgeConfig(label="𝑥₁", show_on="first")
    top = esch.EdgeConfig(label=f"Neurons over data ({task})", show_on=[1])
    edge = esch.EdgeConfigs(left=left, bottom=bottom, top=top)
    path = f"paper/figs/neurs_{cfg.p}_{task}"
    esch.mesh(neurs[1:4, : slice - 8, : slice - 8], edge=edge, font_size=28, path=f"{path}_three.svg")
    esch.mesh(neurs[42, : slice - 8, : slice - 8], edge=edge, font_size=28, path=f"{path}_one.svg")
    left = esch.EdgeConfig(label="ω₀", show_on="first")
    bottom = esch.EdgeConfig(label="ω₁", show_on="first")
    top = esch.EdgeConfig(label=f"Neurons in Fourier space ({task})", show_on=[1])
    edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
    path = f"paper/figs/neurs_{cfg.p}_{task}_fft"
    esch.mesh(
        fft.rfft2(neurs[1:4, :slice, :slice])[:, 1 : 1 + slice // 2, 1:],
        edge=edge,
        font_size=20,
        path=f"{path}_three.svg",
    )
    esch.mesh(
        fft.rfft2(neurs[42, :slice, :slice])[1 : 1 + slice // 2, 1:], edge=edge, font_size=20, path=f"{path}_one.svg"
    )


def plot_grad_norms(scope, cfg, name):
    leafs, struct = tree.flatten(scope.grad_norms)
    ticks = [(i, w) for i, w in enumerate("e.p e.t a.k a.o a.q a.v w.i w.o e.u".split())]
    right = esch.EdgeConfig(ticks=ticks, show_on="all")  # type: ignore
    bottom = esch.EdgeConfig(ticks=[(0, "1"), (49, str(cfg.epochs))], show_on="all", label="Time (linear)")
    left = esch.EdgeConfig(label="Gradient Norm (L2)", show_on="all")
    data = jnp.array(leafs)[:, 1000 :: cfg.epochs // 50]
    data = data / data.max(axis=1, keepdims=True)
    # data = data / data.sum(axis=0, keepdims=True)
    data = data[[4, 5, 0, 1, 2, 3, 6, 7, 8], :]
    top = esch.EdgeConfig(label="Gradient L2 norms for different weight parameters", show_on="all")
    edge = esch.EdgeConfigs(right=right, left=left, bottom=bottom, top=top)
    esch.mesh(data, edge=edge, path=f"paper/figs/grads_norms_{name}.svg", font_size=10)
    # struct


def omega_series_fn(freqs, fname, log_scale=False):
    # neuron_freqs = omega_aux(neuron_freqs)

    # right = esch.EdgeConfig(label="Time", show_on="all")
    left = esch.EdgeConfig(label="{ω}", show_on="all")
    # right = esch.EdgeConfig(ticks=[(0, "0"), (1, "cos(1"), (2, "2")], show_on="all")
    top = esch.EdgeConfig(label="Evolution of active frequencies (ω) through time (log)", show_on="all")
    # bottom = esch.EdgeConfig(label=label_bottom, show_on="all")
    edge = esch.EdgeConfigs(left=left, top=top)
    data = freqs**2
    esch.mesh(data / data.max(1)[:, None], path=f"paper/{fname}.svg", edge=edge, font_size=24)


def fourier_analysis(matrix):
    # Compute 2D FFT
    fft_2d = fft.rfft2(matrix.T).T
    magnitude_spectrum = jnp.abs(fft_2d)
    magnitude_spectrum_centered = fft.fftshift(magnitude_spectrum)
    freq_activations = jnp.linalg.norm(magnitude_spectrum_centered, axis=1)
    significant_freqs = freq_activations > freq_activations.mean() + freq_activations.std()
    return magnitude_spectrum_centered, freq_activations, significant_freqs


def emb_fourier_plots(m, f, s, name):
    # this is the full plot
    top = esch.EdgeConfig(label="Embeddings in Fourier basis", show_on="all")
    bottom = esch.EdgeConfig(label="Token", show_on="all")
    left = esch.EdgeConfig(label="Fourier basis", show_on="all")
    edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
    esch.mesh(m, path=f"paper/figs/fourier_{name}_m.svg", edge=edge, font_size=28)

    # this is the line plot
    ticks_bottom = [(i.item(), f"cos {i // 2}") for i in jnp.where(s)[0] if i % 2 == 1]
    ticks_top = [(0, "const")] + [(i.item(), f"sin {i // 2}") for i in jnp.where(s)[0] if i % 2 == 0]
    top = esch.EdgeConfig(ticks=ticks_top, show_on="all")  # type: ignore
    bottom = esch.EdgeConfig(ticks=ticks_bottom, show_on="all")
    edge = esch.EdgeConfigs(top=top, bottom=bottom)
    if name != "r":
        esch.mesh(f[None, :] ** 2, path=f"paper/figs/fourier_{name}_f.svg", edge=edge, font_size=8)
    else:
        esch.mesh(f[None, :] ** 2, path=f"paper/figs/fourier_{name}_f.svg")


def omega_aux(freqs, kernel_size=3, log_scale=False):
    print(freqs.shape)
    length = (freqs.shape[1] - 1) * 3
    epochs = freqs.shape[0]
    # kernel_size = epochs // length
    conv = lambda row: jnp.convolve(row, jnp.ones(kernel_size) / kernel_size, mode="valid")  # noqa
    freq_series = vmap(conv)(jnp.abs(freqs).T)  # smooth this stuff
    if log_scale:
        freq_series = mi.plots.log_axis_array(freq_series.T, length)
    else:
        freq_series = freq_series[1:, :: epochs // length][..., :length]
    freq_series /= freq_series.sum(axis=1, keepdims=True)

    freq_variance = freq_series.var(axis=0)

    freq_active = (freq_series > freq_series.mean() + freq_series.std()).sum(0)
    # (freq_series > (freq_series.mean() + 1 * freq_series.std())).sum(0)  # noqa
    # print(freq_active)

    # return the line as well
    return freq_series, freq_variance, freq_active


def finding_fn(scope, cfg, task):
    m, variance, active = omega_aux(scope.neuron_freqs[:, 0], log_scale=True)
    # omega_series_fn(, "Time", "", fname="omega-series-1")
    omega_series_fn(m, fname=f"figs/{task}_large_finding")
    # tmp = m / m.max(0, keepdims=True)
    # tmp = (tmp > (tmp.mean(0, keepdims=True) + tmp.std(0, keepdims=True))).astype(float).sum(0, keepdims=True) ** 2
    tmp = active[None, :] ** 1.5

    left = esch.EdgeConfig(label="|{ω}|", show_on=[0])
    bottom = esch.EdgeConfig(label="Time", show_on="all", ticks=[(1, "1"), (56 * 3 - 2, str(cfg.epochs))])
    top = esch.EdgeConfig(
        ticks=[(i, str(int((tmp.squeeze()[i] ** 0.5).item()))) for i in range(1, 56 * 3, 10)],
        show_on="first",
    )
    edge = esch.EdgeConfigs(left=left, bottom=bottom, top=top)
    esch.mesh(
        tmp,
        path=f"paper/figs/{task}_small_finding.svg",
        edge=edge,
        font_size=22,
    )


def wei_plot(acts, cfg, task):
    wei = rearrange(acts.wei[:, 0, :, -1, 0], "(x0 x1) h -> h x0 x1", x0=cfg.p, x1=cfg.p)
    wei = wei[:, :slice, :slice]
    top = esch.EdgeConfig(
        label=[f"Head {i + 1}" for i in range(wei.shape[0])] if task != "nanda" else "", show_on="all"
    )
    left = esch.EdgeConfig(label="𝑥₀", show_on="first")
    right = esch.EdgeConfig(label=task, show_on="last")
    bottom = esch.EdgeConfig(label="𝑥₁", show_on="first")
    edge = esch.EdgeConfigs(left=left, bottom=bottom, top=top, right=right)
    esch.mesh(wei, edge=edge, path=f"paper/figs/{task}_wei.svg", font_size=28)


def final_epoch_neuron_freq(acts):
    pass


# wei_plot(data[nanda_hash][-1], data[nanda_hash][3], "nanda")


# %% work space #################################################################
def plot_hash(hash, name):
    state, metrics, scope, cfg, ds, task, apply, x, acts = data[hash]
    plot_neurs(acts.ffwd, cfg, name)
    emb_fourier_plots(*fourier_analysis(state.params.embeds.tok_emb[:-1]), name)  # type: ignore
    emb_svd(state.params, cfg, name)  # type: ignore
    wei_plot(acts, cfg, name)
    if name not in ["basis", "nanda", "nodro"]:
        plot_grad_norms(scope, cfg, name)
        finding_fn(scope, cfg, name)
    if name not in ["nanda"]:
        mi.plots.plot_run(metrics, ds, cfg, task, hash, font_size=16, log_axis=True)
        pass


# plot_hash(miiii_hash, "miiii")
# plot_hash(masks_hash, "masks")
# plot_hash(basis_hash, "basis")
# plot_hash(nanda_hash, "nanda")
# plot_hash(nodro_hash, "nodro")

# %% Positional embeddings analysis
miiii_pos_emb = data[miiii_hash][0].params.embeds.pos_emb[:2][:, :slice]
nanda_pos_emb = data[nanda_hash][0].params.embeds.pos_emb[:2][:, :slice]  # TODO: THIS SHOLD BE NANDA
pos_emb = jnp.stack((nanda_pos_emb, miiii_pos_emb), axis=0)
label = f"First {slice} dimensions of position embeddings for the factors (top) and prime (bottom) tasks"
left = esch.EdgeConfig(label=["nanda", "miiii"], show_on="all")
top = esch.EdgeConfig(label="Positional embeddings", show_on=[0])
edge = esch.EdgeConfigs(left=left, top=top)
esch.mesh(pos_emb, edge=edge, path="paper/figs/pos_emb.svg", font_size=12)


# %% Model independent plots ######################################################
_cfg = mi.utils.Conf(p=11)
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, _cfg, "remainder", "factors")
x = jnp.concat((ds.x.train, ds.x.eval), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y.train, ds.y.eval), axis=0)[ds.idxs.argsort()]
left = esch.EdgeConfig(label="𝑥₀", show_on="first")
bottom = esch.EdgeConfig(label="𝑥₁", show_on=[5])
top = esch.EdgeConfig(label="Representation of {(𝑥₀, 𝑥₁)} in base-11", show_on=[5])
edge = esch.EdgeConfigs(left=left, bottom=bottom, top=top)
tmp = rearrange(x[:, :2], "(x1 x0) seq ->  x0 x1 seq ", x0=_cfg.p, x1=_cfg.p)
esch.mesh(tmp, edge=edge, path="paper/figs/x_11_plot.svg", font_size=14)


# %% Y plots
nanda_cfg = mi.utils.Conf(p=11)
nanda_ds, _ = mi.tasks.task_fn(random.PRNGKey(0), nanda_cfg, "remainder", "prime")
nanda_y = jnp.concat((nanda_ds.y.train, nanda_ds.y.eval), axis=0)[nanda_ds.idxs.argsort()].reshape(
    (nanda_cfg.p, nanda_cfg.p)
)
primes = jnp.array(oeis["A000040"][1 : y.shape[1] + 1])
bottom = esch.EdgeConfig(label=[f"𝑥 mod {factor}" for factor in primes] + ["𝑥 mod 𝑝"], show_on="all")
top = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(11)], show_on="first")
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(11)], show_on="first")
edge = esch.EdgeConfigs(top=top, left=left, bottom=bottom)
_data = jnp.concat((rearrange(y, "(x0 x1) task ->  task x0 x1 ", x0=11, x1=11), nanda_y[None, ...]), axis=0)
# data /= data.max(axis=(1, 2))[:, None, None]
esch.mesh(_data, edge=edge, path="paper/figs/y_11_plot.svg", font_size=13)


# %% Polar Plots
primes = jnp.array(oeis["A000040"][1:1000])
ps = jnp.array(primes[primes < (113**2)])
_11s = jnp.arange(0, 113**2, 11)
_7_23 = jnp.concat((jnp.arange(0, 113**2, 13), jnp.arange(0, 113**2, 23)))
plt.style.use("default")
mi.plots.small_multiples(fnames=["n", "t", "n"], seqs=[_7_23, _11s, ps], f_name="polar", n_rows=1, n_cols=3)
# plt.close()

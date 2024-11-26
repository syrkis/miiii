# %% notebook.py
#   miiii notebook
# by: Noah Syrkis


# %% Imports
import esch
import jax.numpy as jnp
from jax import random, tree
from jax.numpy import fft
from functools import partial
from einops import rearrange
# import matplotlib.pyplot as plt

import miiii as mi

# %% constants
rng = random.PRNGKey(0)
slice = 37

# %% F task load
f_hash = "441149ef948942aca4e8c391"
f_state, f_metrics, f_scope, f_cfg = mi.utils.get_metrics_and_params(f_hash)
f_ds, f_task = mi.tasks.task_fn(rng, f_cfg, "remainder", "factors")
f_apply = partial(mi.model.apply_fn(f_cfg, f_ds, f_task, False), random.PRNGKey(0))
f_x = jnp.concat((f_ds.x.train, f_ds.x.eval))[f_ds.idxs.argsort()]
f_acts = f_apply(f_state.params, f_x)

# %% p task
p_hash = "0c848c1444264cbfa1a4de6e"
p_state, p_metrics, p_cfg = mi.utils.get_metrics_and_params(p_hash, task_span="prime")
p_ds, p_task = mi.tasks.task_fn(rng, p_cfg, "remainder", "prime")
p_apply = partial(mi.model.apply_fn(p_cfg, p_ds, p_task, False), random.PRNGKey(0))
p_x = jnp.concat((p_ds.x.train, p_ds.x.eval))[p_ds.idxs.argsort()]
p_acts = p_apply(p_state.params, p_x)


# %% Positional embeddings analysis
f_pos_emb = f_state.params.embeds.pos_emb[:2][:, :slice]
p_pos_emb = p_state.params.embeds.pos_emb[:2][:, :slice]
pos_emb = jnp.stack((p_pos_emb, f_pos_emb), axis=0)
label = f"First {slice} dimensions of position embeddings for the factors (top) and prime (bottom) tasks"
left = esch.EdgeConfig(label=["nanda", "miiii"], show_on="all")
edge = esch.EdgeConfigs(left=left)
esch.plot(pos_emb, edge=edge, path="paper/figs/pos_emb.svg")


# %% Token embedding exploratoray analysis
f_tok_emb = f_state.params.embeds.tok_emb[: f_cfg.p]
f_U, f_S, f_V = jnp.linalg.svd(f_tok_emb)
p_tok_emb = p_state.params.embeds.tok_emb[: p_cfg.p]
p_U, p_S, p_V = jnp.linalg.svd(p_tok_emb)

f_S_50 = jnp.where((f_S / f_S.sum()).cumsum() < 0.5)[0].max()
p_S_50 = jnp.where((p_S / p_S.sum()).cumsum() < 0.5)[0].max()
f_S_90 = jnp.where((f_S / f_S.sum()).cumsum() < 0.9)[0].max()
p_S_90 = jnp.where((p_S / p_S.sum()).cumsum() < 0.9)[0].max()
S = jnp.stack((p_S / p_S.sum(), f_S / f_S.sum()), axis=0).reshape((2, 1, -1))[:, :, :83]

top = esch.EdgeConfig(ticks=[(f_S_50.item(), "0.5"), (f_S_90.item(), "0.9")], show_on="first")
left = esch.EdgeConfig(label=["nanda", "miiii"], show_on="all")
bottom = esch.EdgeConfig(ticks=[(p_S_50.item(), "0.5"), (p_S_90.item(), "0.9")], show_on="last")
edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
esch.plot(S, edge=edge, path="paper/figs/S.svg")


esch.plot(f_U[:, : f_S_50.item()].T, path="paper/figs/f_U.svg")
esch.plot(p_U[:, : p_S_50.item()].T, path="paper/figs/p_U.svg")
# %% plots
mi.plots.plot_run(f_metrics, f_ds, f_cfg, f_task, f_hash)


# %% Embeddings fourier analsysis
p_m, p_f, p_s = mi.plots.fourier_analysis(p_state.params.embeds.tok_emb[:-1])
f_m, f_f, f_s = mi.plots.fourier_analysis(f_state.params.embeds.tok_emb[:-1])
r_m, r_f, r_s = mi.plots.fourier_analysis(mi.model.initializer(rng, f_state.params.embeds.tok_emb[:-1].shape))


# %%
def fourier_plots(m, f, s, name):
    esch.plot(m, path=f"paper/figs/fourier_{name}_m.svg")
    ticks_bottom = [(i.item(), f"cos {i//2}") for i in jnp.where(s)[0] if i % 2 == 1]
    ticks_top = [(0, "constant")] + [(i.item(), f"sin {i//2}") for i in jnp.where(s)[0] if i % 2 == 0]
    top = esch.EdgeConfig(ticks=ticks_top, show_on="all")  # type: ignore
    bottom = esch.EdgeConfig(ticks=ticks_bottom, show_on="all")
    edge = esch.EdgeConfigs(top=top, bottom=bottom)
    if name != "r":
        esch.plot(f[None, :], path=f"paper/figs/fourier_{name}_f.svg", edge=edge)
    else:
        esch.plot(f[None, :], path=f"paper/figs/fourier_{name}_f.svg")


fourier_plots(p_m, p_f, p_s, "p")
fourier_plots(f_m, f_f, f_s, "f")
fourier_plots(r_m, r_f, r_s, "r")

# %%
fourier_basis = fft.rfft2(jnp.eye(p_cfg.p))
esch.plot((fourier_basis.T @ fourier_basis.conj()).imag)
# fourier_basis

# %%
heads = rearrange(p_acts.wei, "(a b) l h x0 x1 -> h a b l x0 x1", a=p_cfg.p, b=p_cfg.p).squeeze()[
    :, :slice, :slice, -1, 0
]
esch.plot(heads[0])
# p_acts.wei.shape


# %%
neurons = rearrange(f_acts.ffwd, "(x0 x1) l p n -> n x0 x1 l p", x0=p_cfg.p, x1=p_cfg.p).squeeze()[..., -1]
top = esch.EdgeConfig(ticks=[(0, "0"), (36, "36")], show_on="all", label="𝑥₀")
left = esch.EdgeConfig(ticks=[(0, "0"), (36, "36")], show_on="all", label="𝑥₁")
edge = esch.EdgeConfigs(top=top, left=left)
esch.plot(heads[0], edge=edge, path="paper/figs/plot_intro.svg")

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
leafs, struct = tree.flatten(f_metrics.grads)
ticks = [(i, w) for i, w in enumerate("emb_pos emb_tok k o q v w_in w_out unbeds".split())]
right = esch.EdgeConfig(ticks=ticks, show_on="all")  # type: ignore
top = esch.EdgeConfig(ticks=[(0, "1"), (49, str(f_cfg.epochs))], show_on="all", label="Time (linear)")
left = esch.EdgeConfig(label="Gradient Norm (L2)", show_on="all")
edge = esch.EdgeConfigs(right=right, left=left, top=top)
data = jnp.array(leafs)[:, 1000 :: f_cfg.epochs // 50]
data = data / data.max(axis=1, keepdims=True)
# data = data / data.sum(axis=0, keepdims=True)
data = data[[4, 5, 0, 1, 2, 3, 6, 7, 8], :]
esch.plot(data, edge=edge, path=f"paper/figs/grads_norms.svg")
# struct

# %%
dft = jnp.fft.fft(jnp.eye(p_cfg.p))
esch.plot(dft.real, path="paper/figs/real_dft.svg")
# dft = dft / jnp.linalg.norm(dft, axis=1)
esch.plot(dft @ f_state.params.embeds.tok_emb[:-1])


# %%
esch.plot(jnp.abs(dft[f_s.repeat(2)[1:]][::2].mean(0))[None, :])


# %%  W_NEUR
f_neurs = rearrange(p_acts.ffwd.squeeze()[:, -1], "(x0 x1) n -> n x0 x1", x0=f_cfg.p, x1=f_cfg.p)
esch.plot(dft @ f_neurs[0] @ dft.T)

# %% Neuron frac explained by Freq
neuron_freq_norm = jnp.zeros((f_cfg.p // 2, p_cfg.latent_dim * 4))
for freq in range(0, p_cfg.p // 2):
    for x in [0, 2 * freq - 1, 2 * freq]:
        for y in [0, 2 * freq - 1, 2 * freq]:
            tmp = neuron_freq_norm[freq] + f_neurs[:, x, y] ** 2
            neuron_freq_norm.at[freq].set(tmp)
neuron_freq_norm = neuron_freq_norm / (f_neurs**2).sum(axis=(-1, -2), keepdims=True)

# %%
# jnp.unique(neuron_freq_norm)
#
# esch.plot((dft @ f_neurs @ dft.T)[1])

esch.plot(jnp.abs(fft.rfft2(f_neurs).mean((0, 1)))[None, 1:])  # THIS IS INTERSTING
f_neurs.shape, fft.rfft2(f_neurs).shape
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
rearrange(p_acts.logits, "(a b) c -> a b c", a=p_cfg.p, b=p_cfg.p).shape

# %%
f_acts.ffwd.squeeze().shape, f_neurs.shape


# %%
esch.plot(f_scope.neuron_freqs[:: f_cfg.epochs // 200, 1:].T)

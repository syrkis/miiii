# %% notebook.py
#   miiii notebook
# by: Noah Syrkis


# %% Imports
import esch
import jax.numpy as jnp
from jax import random

import miiii as mi

# %% constants
rng = random.PRNGKey(0)
slice = 37

# %% F task load
f_hash = "4a98603ba79c4ed2895f9670"
f_state, f_metrics, f_cfg = mi.utils.get_metrics_and_params(f_hash)
f_ds, f_task = mi.tasks.task_fn(rng, f_cfg, "remainder", "factors")

# %% p task
p_hash = "0c848c1444264cbfa1a4de6e"
p_state, p_metrics, p_cfg = mi.utils.get_metrics_and_params(p_hash, task_span="prime")
p_ds, p_task = mi.tasks.task_fn(rng, p_cfg, "remainder", "prime")


# %% Positional embeddings analysis
f_pos_emb = f_state.params.embeds.pos_emb[:2][:, :slice]
p_pos_emb = p_state.params.embeds.pos_emb[:2][:, :slice]
pos_emb = jnp.stack((f_pos_emb, p_pos_emb), axis=0)
label = f"First {slice} dimensions of position embeddings for the factors (top) and prime (bottom) tasks"
left = esch.EdgeConfig(label=["𝑓-task", "𝑝-task"], show_on="all")
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
S = jnp.stack((f_S / f_S.sum(), p_S / p_S.sum()), axis=0).reshape((2, 1, -1))[:, :, :83]

top = esch.EdgeConfig(ticks=[(f_S_50.item(), "0.5"), (f_S_90.item(), "0.9")], show_on="first")
left = esch.EdgeConfig(label=["𝑓-task", "𝑝-task"], show_on="all")
bottom = esch.EdgeConfig(ticks=[(p_S_50.item(), "0.5"), (p_S_90.item(), "0.9")], show_on="last")
edge = esch.EdgeConfigs(top=top, bottom=bottom, left=left)
esch.plot(S, edge=edge, path="paper/figs/S.svg")


esch.plot(f_U[:, : f_S_50.item()].T, path="paper/figs/f_U.svg")
esch.plot(p_U[:, : p_S_50.item()].T, path="paper/figs/p_U.svg")
# %% plots
mi.plots.plot_run(f_metrics, f_ds, f_cfg, f_task, f_hash)


# %% Embeddings fourier analsysis
# for e in (
# p_state.params.embeds.tok_emb[:-1],
# f_state.params.embeds.tok_emb[:-1],
# mi.model.initializer(rng, f_state.params.embeds.tok_emb[:-1].shape),
# ):
p_m, p_f, p_s = mi.plots.fourier_analysis(p_state.params.embeds.tok_emb[:-1])
f_m, f_f, f_s = mi.plots.fourier_analysis(f_state.params.embeds.tok_emb[:-1])
r_m, r_f, r_s = mi.plots.fourier_analysis(mi.model.initializer(rng, f_state.params.embeds.tok_emb[:-1].shape))

esch.plot(p_m[p_m.shape[0] // 2 :], path="paper/figs/fourier_p_m.svg")
esch.plot(p_f[None, :], path="paper/figs/fourier_p_f.svg")
esch.plot(f_m, path="paper/figs/fourier_f_m.svg")
esch.plot(f_f[None, :], path="paper/figs/fourier_f_f.svg")
esch.plot(r_m, path="paper/figs/fourier_r_m.svg")
esch.plot(r_f[None, :], path="paper/figs/fourier_r_f.svg")


# %%

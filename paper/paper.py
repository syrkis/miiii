# %% paper.py
#   generates plots for paper
# by: Noah Syrkis

# """
# This script generates plots for the paper.
# Specifically, we need:
#     - X plots for p=11
#     - Y plots for p=11
#         - remainder and divisibility tasks
#     - Training "curves" for the "final" model.
#     - Generalization vector of final train step (train and valid above each other).
# """

# %% Imports
import miiii as mi

import esch
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange
from jax import random
from oeis import oeis


# Diagrams and illustrations
# %% X plots
cfg = mi.utils.Conf(p=11)
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
x = jnp.concat((ds.x_train, ds.x_valid), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y_train, ds.y_valid), axis=0)[ds.idxs.argsort()]
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
bottom = esch.EdgeConfig(label=[str(i) for i in range(cfg.p)], show_on="all")
edge = esch.EdgeConfigs(left=left, bottom=bottom)
esch.plot(rearrange(x[:, :2], "(x1 x0) seq ->  x0 x1 seq ", x0=cfg.p, x1=cfg.p), edge=edge, path="figs/x_11_plot.svg")

# %% Y plots
nanda_cfg = mi.utils.Conf(p=11)
nanda_ds, _ = mi.tasks.task_fn(random.PRNGKey(0), nanda_cfg, "remainder", "prime")
nanda_y = jnp.concat((nanda_ds.y_train, nanda_ds.y_valid), axis=0)[nanda_ds.idxs.argsort()].reshape(
    (nanda_cfg.p, nanda_cfg.p)
)
primes = jnp.array(oeis["A000040"][1 : y.shape[1] + 1])
bottom = esch.EdgeConfig(label=[f"Factor {factor} remainder" for factor in primes] + ["Prime remainder"], show_on="all")
top = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
edge = esch.EdgeConfigs(top=top, left=left, bottom=bottom)
data = jnp.concat((rearrange(y, "(x0 x1) task ->  task x0 x1 ", x0=cfg.p, x1=cfg.p), nanda_y[None, ...]), axis=0)
# data /= data.max(axis=(1, 2))[:, None, None]
esch.plot(data, edge=edge, path="figs/y_11_plot.svg")


# %%


# %% Polar Plots
primes = jnp.array(oeis["A000040"][1:1000])
ps = jnp.array(primes[primes < (113**2)])
_11s = jnp.arange(0, 113**2, 11)
_7_23 = jnp.concat((jnp.arange(0, 113**2, 7), jnp.arange(0, 113**2, 23)))
plt.style.use("default")
mi.plots.small_multiples(fnames=["n", "t", "n"], seqs=[_7_23, _11s, ps], f_name="polar", n_rows=1, n_cols=3)
# remove plot
# plt.close()


# %% Constants
hash = "33214cc96bff4aa1a0f8d532"
slice = 37
(state, metrics, cfg), _ = mi.utils.get_metrics_and_params(hash)  # get a run
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
x = jnp.concat((ds.x_train, ds.x_valid), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y_train, ds.y_valid), axis=0)[ds.idxs.argsort()]


# %% ATTENTION WEIGHTS
apply = mi.model.apply_fn(cfg, ds, task, eval=True)
acts = apply(rng, state.params, x)
esch.plot(
    rearrange(acts.wei.squeeze(), "(x0 x1) heads from to ->  heads x0 x1 from to", x0=cfg.p, x1=cfg.p)[:, :, :, -1, 0][
        :, :slice, :slice
    ]
)

# %% EMBEDDINGS
U, S, V = jnp.linalg.svd(state.params.embeds.tok_emb[: cfg.p])
s = (S / S.sum()).cumsum()
_5 = jnp.where(s < 0.5)[0].max()
_9 = jnp.where(s <= 0.9)[0].max()
_99 = jnp.where(s <= 0.99)[0].max()


# %% singular values
bottom = esch.EdgeConfig(ticks=[(_5.item(), "0.5"), (_9.item() - 1, "0.9"), (_99.item() - 1, "0.99")], show_on="first")
left = esch.EdgeConfig(ticks=[(0, "S")], show_on="first")
edge = esch.EdgeConfigs(bottom=bottom, left=left)
esch.plot(S[None, :_9], edge=edge)

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
esch.plot(metrics.train.acc[:: cfg.epochs // 100].T)

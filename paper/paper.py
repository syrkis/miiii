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
x = jnp.concat((ds.x.train, ds.x.eval, ds.x.test), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y.train, ds.y.eval, ds.y.test), axis=0)[ds.idxs.argsort()]
left = esch.EdgeConfig(ticks=[(i, str(i)) for i in range(cfg.p)], show_on="first")
bottom = esch.EdgeConfig(label=[str(i) for i in range(cfg.p)], show_on="all")
edge = esch.EdgeConfigs(left=left, bottom=bottom)
esch.plot(rearrange(x[:, :2], "(x1 x0) seq ->  x0 x1 seq ", x0=cfg.p, x1=cfg.p), edge=edge, path="figs/x_11_plot.svg")

# %% Y plots
nanda_cfg = mi.utils.Conf(p=11)
nanda_ds, _ = mi.tasks.task_fn(random.PRNGKey(0), nanda_cfg, "remainder", "prime")
nanda_y = jnp.concat((nanda_ds.y.train, nanda_ds.y.eval, nanda_ds.y.test), axis=0)[nanda_ds.idxs.argsort()].reshape(
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
hash = "f0789b8a88a343bd813abcb8"
slice = 37
state, metrics, cfg = mi.utils.get_metrics_and_params(hash, "factors")  # get a run
rng = random.PRNGKey(0)
ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")
x = jnp.concat((ds.x.train, ds.x.eval, ds.x.test), axis=0)[ds.idxs.argsort()]
y = jnp.concat((ds.y.train, ds.y.eval, ds.y.test), axis=0)[ds.idxs.argsort()]


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
_77 = jnp.where(s < 0.77)[0].max()
_9 = jnp.where(s <= 0.9)[0].max()
_99 = jnp.where(s <= 0.99)[0].max()

# %% singular values
bottom = esch.EdgeConfig(
    ticks=[(_5.item(), "0.5"), (_9.item() - 1, "0.9"), (_77.item() - 1, "0.77"), (_99.item() - 1, "0.99")],
    show_on="first",
)
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
left = esch.EdgeConfig(
    ticks=[(0, "2"), (metrics.train.loss.shape[1] - 1, str(task.primes.max()))], show_on="first", label="Task"
)
top = esch.EdgeConfig(ticks=[(0, "1"), (99, str(cfg.epochs))], show_on="first", label="Time")
edge = esch.EdgeConfigs(left=left, top=top)
esch.plot(metrics.valid.acc[:: cfg.epochs // 100].T, edge=edge, path="figs/valid_acc.svg")
esch.plot(metrics.train.acc[:: cfg.epochs // 100].T, edge=edge, path="figs/train_acc.svg")

# %%
# U, S, V = jnp.linalg.svd(state.params.embeds.pos_emb)
# U.round(1)
tmp = state.params.embeds.pos_emb.round(3)


# %%
# pickle link

esch.plot(state.params.embeds.pos_emb.round(3))

tmp = jnp.array(
    [
        [
            -0.008,
            0.01,
            -0.009,
            0.014,
            0.011,
            -0.005,
            -0.016,
            -0.005,
            -0.001,
            -0.004,
            -0.001,
            0.026,
            0.002,
            0.002,
            0.007,
            -0.011,
            0.002,
            -0.006,
            -0.024,
            0.009,
            0.005,
            -0.01,
            -0.004,
            -0.001,
            -0.024,
            0.006,
            -0.003,
            -0.002,
            0.007,
            -0.004,
            -0.006,
            -0.001,
            -0.001,
            0.002,
            -0.024,
            -0.005,
            -0.001,
            0.011,
            0.003,
            0.003,
            0.002,
            0.007,
            0.002,
            -0.018,
            0.009,
            -0.027,
            0.003,
            0.017,
            0.002,
            -0.02,
            -0.01,
            -0.009,
            0.006,
            -0.01,
            -0.001,
            0.017,
            -0.0,
            -0.012,
            -0.008,
            0.015,
            -0.006,
            0.001,
            -0.012,
            0.014,
            0.026,
            -0.001,
            -0.01,
            0.002,
            -0.017,
            0.002,
            0.0,
            -0.007,
            -0.015,
            0.0,
            0.014,
            -0.011,
            0.0,
            -0.005,
            0.007,
            -0.008,
            0.003,
            -0.0,
            0.004,
            0.007,
            0.0,
            0.002,
            0.001,
            0.0,
            -0.001,
            -0.01,
            0.007,
            0.001,
            -0.0,
            0.006,
            -0.002,
            -0.011,
            -0.035,
            -0.003,
            -0.003,
            -0.0,
            -0.02,
            -0.004,
            -0.008,
            0.0,
            0.001,
            -0.015,
            0.0,
            0.001,
            -0.021,
            -0.005,
            -0.006,
            -0.022,
            -0.006,
            0.001,
            0.003,
            -0.005,
            0.005,
            -0.003,
            0.013,
            -0.0,
            0.014,
            -0.009,
            -0.003,
            0.001,
            -0.001,
            0.004,
            -0.017,
            0.005,
        ],
        [
            -0.009,
            0.011,
            -0.009,
            0.015,
            0.012,
            -0.004,
            -0.016,
            -0.006,
            0.0,
            -0.005,
            -0.002,
            0.025,
            0.001,
            0.003,
            0.007,
            -0.013,
            0.003,
            -0.006,
            -0.025,
            0.008,
            0.004,
            -0.01,
            -0.003,
            -0.001,
            -0.024,
            0.006,
            -0.003,
            -0.001,
            0.006,
            -0.002,
            -0.006,
            -0.001,
            -0.0,
            0.002,
            -0.022,
            -0.004,
            -0.002,
            0.01,
            0.002,
            0.002,
            0.001,
            0.007,
            0.002,
            -0.019,
            0.008,
            -0.028,
            0.003,
            0.02,
            0.0,
            -0.021,
            -0.012,
            -0.009,
            0.007,
            -0.012,
            0.0,
            0.016,
            -0.001,
            -0.011,
            -0.008,
            0.015,
            -0.005,
            0.001,
            -0.011,
            0.014,
            0.027,
            0.0,
            -0.008,
            0.002,
            -0.019,
            0.002,
            0.002,
            -0.006,
            -0.014,
            -0.001,
            0.014,
            -0.012,
            0.0,
            -0.004,
            0.007,
            -0.009,
            0.002,
            -0.001,
            0.003,
            0.006,
            0.002,
            0.0,
            0.0,
            -0.0,
            -0.002,
            -0.008,
            0.008,
            0.0,
            0.001,
            0.005,
            -0.003,
            -0.012,
            -0.034,
            -0.003,
            -0.004,
            0.001,
            -0.021,
            -0.005,
            -0.009,
            -0.001,
            0.002,
            -0.016,
            0.0,
            -0.0,
            -0.019,
            -0.006,
            -0.006,
            -0.022,
            -0.004,
            0.001,
            0.002,
            -0.004,
            0.004,
            -0.004,
            0.011,
            0.001,
            0.015,
            -0.011,
            -0.002,
            0.0,
            -0.001,
            0.005,
            -0.019,
            0.004,
        ],
        [
            0.064,
            -0.045,
            0.053,
            -0.164,
            -0.075,
            -0.031,
            0.176,
            -0.033,
            -0.055,
            0.141,
            0.054,
            0.042,
            -0.077,
            -0.063,
            -0.061,
            0.091,
            0.04,
            -0.14,
            0.169,
            0.051,
            0.06,
            0.049,
            0.153,
            -0.053,
            -0.055,
            -0.042,
            0.065,
            0.143,
            0.052,
            0.069,
            0.047,
            0.065,
            0.067,
            -0.041,
            0.077,
            0.134,
            -0.073,
            -0.211,
            -0.126,
            0.057,
            -0.046,
            -0.073,
            -0.142,
            -0.066,
            -0.204,
            0.046,
            -0.065,
            -0.056,
            -0.059,
            0.097,
            0.057,
            0.076,
            -0.1,
            0.058,
            -0.056,
            -0.144,
            0.06,
            0.108,
            -0.062,
            -0.159,
            0.148,
            -0.066,
            -0.082,
            -0.183,
            -0.07,
            -0.054,
            0.172,
            -0.049,
            0.046,
            -0.07,
            -0.032,
            0.211,
            -0.175,
            0.065,
            0.048,
            0.157,
            -0.07,
            -0.063,
            -0.175,
            0.038,
            0.05,
            0.045,
            -0.054,
            -0.058,
            -0.141,
            -0.051,
            -0.061,
            0.052,
            -0.046,
            0.149,
            -0.049,
            -0.104,
            -0.089,
            -0.147,
            -0.07,
            0.05,
            0.064,
            -0.066,
            0.062,
            -0.071,
            0.05,
            0.053,
            0.059,
            0.066,
            -0.035,
            0.052,
            -0.053,
            0.057,
            -0.066,
            0.166,
            -0.1,
            -0.063,
            0.052,
            -0.055,
            0.122,
            0.15,
            -0.044,
            0.027,
            -0.214,
            0.095,
            -0.173,
            0.054,
            -0.051,
            -0.144,
            -0.095,
            0.047,
            0.116,
            -0.072,
        ],
    ]
)


# %%
left = esch.EdgeConfig(ticks=[(0, "𝑥₀"), (1, "𝑥₁")], show_on="all")
edge = esch.EdgeConfigs(left=left)
esch.plot(
    jnp.stack((tmp[:2, :slice], state.params.embeds.pos_emb[:2, :slice]), axis=0), path="figs/pos_emb.svg", edge=edge
)  # this is an important plot

# %%
pos_emb = state.params.embeds.pos_emb[:2]
esch.plot(pos_emb @ pos_emb.T)  # TODO: put this in a table with the thing

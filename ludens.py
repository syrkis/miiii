# %% ludens.py
#   ludens notebook
# by: Noah Syrkis


# %% Imports
import miiii as mi
import esch
import jax.numpy as jnp
from jax import random
from einops import rearrange
import matplotlib.pyplot as plt


# %%  Load dataset and declare hash, etc.
hash = "c0aa51f06ee54ab280e5c625"
cfg = mi.utils.construct_cfg_from_hash(hash)
rng = random.PRNGKey(0)
ds = mi.tasks.task_fn(rng, cfg)
x, y = map(lambda x, y: jnp.concat((x, y), axis=0)[jnp.argsort(ds.idxs)], ds.train, ds.valid)


# %%  Get metrics and params from s3 bucket
state: mi.train.State
metrics: mi.train.Metrics
acts: mi.model.Activation
state, (metrics, _) = mi.utils.get_metrics_and_params(hash)  # type: ignore
# %%
plt.plot(metrics.train.loss)
plt.plot(metrics.valid.loss)

# %% Get activations
apply = mi.model.apply_fn(cfg)
acts = apply(state.params, rng, x, 0.0)


# %% Exploring attention patterns
esch.plot(acts.wei.squeeze().mean(0)[:, -1, :-1])  # average attention is uniform. ( a and b are euqlly imporant)
esch.plot(acts.wei.squeeze()[7][:, -1, :-1])  #   but it depends on the sample


# %%  Attention for result to and and b for the different heads.
weis = rearrange(acts.wei, "(fst snd) layer head a b -> head fst snd layer a b", fst=cfg.p, snd=cfg.p).squeeze()[
    :, :, :, -1, :-1
]
esch.plot(
    weis[:, :, :, 0],
    edge=esch.EdgeConfigs(bottom=esch.EdgeConfig(label=[f"Head {i}" for i in range(4)], show_on="all")),
    path="weis.svg",
)

# %%
W_E = state.params.embeds.tok_emb[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ state.params.attn.v[0] @ state.params.attn.o[0] @ state.params.ffwd.w_in
print("W_neur", W_neur.shape)
W_logit = state.params.ffwd.w_out[0] @ state.params.unbeds
print("W_logit", W_logit.shape)


# %%
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[0:5], path="ffwd.svg")
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[5:10], path="ffwd.svg")
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[10:15], path="ffwd.svg")
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[15:20], path="ffwd.svg")
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[20:25], path="ffwd.svg")

# %%

# %%
U, S, V = jnp.linalg.svd(state.params.embeds.tok_emb[:-1, :-1])

(S / S.sum()).cumsum()

# %%
esch.plot(
    U, edge=esch.EdgeConfigs(bottom=esch.EdgeConfig(label=f"{U.shape[0]}x{U.shape[1]}", show_on="first")), path="U.svg"
)
esch.plot(S[None, :], path="S.svg")
esch.plot(V, path="V.svg")


# %%

import jax.numpy as np
from jax import random

rng = random.PRNGKey(0)
n = 15
D = np.tril(random.randint(rng, (10, 10), 0, 100)) * (1 - jnp.eye(10))
D = D + D.T

D[np.arange(10), D.argsort(0)[:, 1:].argmax(-1)]

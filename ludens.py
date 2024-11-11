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
hash = "315376a04a0843f0a4fa6f75"
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
# plt.plot(metrics.train.f1)
# plt.plot(metrics.valid.f1)

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
esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[0:4], path="ffwd.svg")

# %%

# %%
U, S, V = jnp.linalg.svd(W_E)

plt.plot(S)
jnp.where((S / S.sum()).cumsum() < 0.9)[0].max()

# %%
esch.plot(U[:, :10].T)
esch.plot(U[:, -10:].T)
# plt.imshow(U[:, :10].T)
#
#
# %%
esch.plot(
    U, edge=esch.EdgeConfigs(bottom=esch.EdgeConfig(label=f"{U.shape[0]}x{U.shape[1]}", show_on="first")), path="U.svg"
)
esch.plot(S[None, :], path="S.svg")
esch.plot(V, path="V.svg")


# %%
fig, ax = plt.subplots(figsize=(30, 5))
ax.plot(U[:, :8])
# %% get fourier basis
F = []
for freq in range(1, cfg.p // 2 + 1):
    F.append(jnp.sin(jnp.arange(cfg.p) * 2 * jnp.pi * freq / cfg.p))
    F.append(jnp.cos(jnp.arange(cfg.p) * 2 * jnp.pi * freq / cfg.p))

F = jnp.stack(F, axis=0)
F = F / jnp.linalg.norm(F, axis=-1, keepdims=True)
esch.plot(F)

# %%
esch.plot(F @ F.T)

# %%
esch.plot(F @ W_E)

# %%
esch.plot((jnp.linalg.norm(F @ W_E, axis=-1))[None, :])

# %%
jnp.linalg.norm(F @ W_E, axis=-1).argsort()
# %%
esch.plot(W_E)

# %%

key_freqs = jnp.array([97, 94, 96, 93, 92, 0, 1])
key_idxs = (key_freqs.repeat(2) + jnp.tile(jnp.eye(2)[1], (len(key_freqs),))).astype(jnp.int32)
key_embed = (F @ W_E)[key_idxs]
esch.plot(key_embed @ key_embed.T)
# %%
# esch.plot(rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)[0], path="ffwd.svg")

# %%
esch.plot(F[40][None, :] * F[40][:, None])

# %%
neuron_acts = rearrange(acts.ffwd.squeeze()[:, -1], "(a b) neuron -> neuron a b", a=cfg.p, b=cfg.p)
esch.plot(F @ neuron_acts[ds.idxs[-1]] @ F.T)
# %%
# W_logit[:, 0] @ F.T
# W_logit.shape, F.shape
esch.plot((state.params.ffwd.w_out[0] @ state.params.unbeds)[-1].T, path="out.svg")

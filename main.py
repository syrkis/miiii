# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiii as mi
import sys
from jax import random
import jax.numpy as jnp
import esch
from einops import rearrange


# %% Configuration
cfg = mi.utils.Conf(project="miiii", p=113, epochs=10000, lamb=2, dropout=0.5, l2=1.0)
rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg)
state, (metrics, _) = mi.train.train(keys[1], cfg, ds)  # scope=True)

# %%
params = mi.model.init_fn(rng, cfg, ds)
x = jnp.concat((ds.train[0], ds.valid[0]), axis=0)[jnp.argsort(ds.idxs)]
y = jnp.concat((ds.train[1], ds.valid[1]), axis=0)[jnp.argsort(ds.idxs)]
apply = mi.model.apply_fn(cfg)
acts = apply(state.params, rng, x, 0.0)

# %%
# %%
# W_E = state.params.embeds.tok_emb
# W_E.shape

# %%
# W_neur = state.params.embeds.tok_emb @ state.params.attn.v[0] @ state.params.attn.o[0] @ state.params.ffwd.w_in[0]
# W_neur.shape

# %%

# %%
# W_logit = state.params.ffwd.w_out[0] @ state.params.unbeds
# W_logit.shape


# %%  we see the attention is leaning towards the first digit (from the right)
# wei = rearrange(acts.wei, " (a b) layer head fst snd ->  a b layer head fst snd", a=cfg.p, b=cfg.p)
# esch.plot(wei.squeeze()[:, :, :, 0, 0].transpose(2, 0, 1))
# esch.plot(wei.squeeze()[:, :, :, 1, 0].transpose(2, 0, 1))
# esch.plot(wei.squeeze()[:, :, :, 0, 1].transpose(2, 0, 1))
# esch.plot(wei.squeeze()[:, :, :, 1, 1].transpose(2, 0, 1))


# %%


# %%  Neural activations (first five mlp neurons)
# %% Logging stuff
# U, S, V = jnp.linalg.svd(W_E)

# %%
# thresh = jnp.where((S / S.sum()).cumsum(axis=0) < 0.95)[0].max()
# esch.plot(S[None, :])  # chose 90th percentile.
# esch.plot(U[:, :thresh].T)
# plt.show()
# %%


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

if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    mi.utils.log_fn(cfg, ds, state, metrics)

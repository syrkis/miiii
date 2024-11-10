# %% ludens.py
#   ludens notebook
# by: Noah Syrkis


# %% Imports
import miiii as mi
import esch
import jax.numpy as jnp


# %%
hash = "de87b3900af64fdeb34bea42"
state: mi.train.State
metrics: mi.train.Metrics
acts: mi.model.Activation
state, (metrics, acts) = mi.utils.get_metrics_and_params(hash)  # type: ignore

# %%
U, S, V = jnp.linalg.svd(state.params.embeds.tok_emb)
esch.plot(U)
esch.plot(S[None, :])

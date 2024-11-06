# %% scope.py
#   neuralnetscopy codebase for visualizing neural networks
# by: Noah Syrkis

# %% imports
from jax import tree
from miiiii.train import State
import esch


# %%
def scope_fn(state):
    tok_emb = state.params.embeds.tok_emb
    pos_emb = state.params.embeds.pos_emb

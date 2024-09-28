# %% scope.py
#   neuralnetscopy codebase for visualizing neural networks
# by: Noah Syrkis

# %% imports
import jax
from jax import random, grad, jit, value_and_grad, tree_util
import optax
import pickle
from functools import partial
from chex import dataclass, Array


@dataclass
class Activation:
    q: Array
    k: Array
    v: Array
    qk: Array
    wei: Array
    weiv: Array


# %%
def scope_fn(params, x):
    flat_params, tree = tree_util.tree_flatten(params)
    print(tree)

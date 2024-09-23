# %% scope.py
#   neuralnetscopy codebase for visualizing neural networks
# by: Noah Syrkis

# %% imports
import miiiii as mi
import jax
from jax import random, grad, jit, value_and_grad, tree_util
import optax
import pickle


params = mi.utils.load_params("model.pkl")


# %%


def scope_fn(params, x):
    flat_params, tree = tree_util.tree_flatten(params)
    print(tree)

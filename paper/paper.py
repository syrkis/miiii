# %% paper.py
#   generates plots for paper
# by: Noah Syrkis

"""
This script generates plots for the paper.
Specifically, we need:
    - X plots for p=11
    - Y plots for p=11
        - remainder and divisibility tasks
    - Training "curves" for the "final" model.
    - Generalization vector of final train step (train and valid above each other).
"""

# %% Imports
import miiii as mi
import jax.numpy as jnp
from jax import random

# %% Constants

# %% Pass

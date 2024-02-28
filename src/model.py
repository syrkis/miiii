# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

from functools import partial
from tqdm import tqdm
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple, Union

from src.data import decode

# functions
def embed_fn(params, x):
    n  = x.shape[1]                              # num toks in sample
    x  = params['tok_embedding'][x]              # tok embeddings
    x += params['pos_embedding'][jnp.arange(n)]  # pos embeddings
    return x
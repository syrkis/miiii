# stats.py
#   stats functions for miiiii
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import Array
from einops import repeat


# functions
def horizontal_mean_pooling(x: Array, width: int = 3) -> Array:
    """Rolling mean array. Shrink to be rows x rows * width."""
    x = x[:, : (x.shape[1] // (x.shape[0] * width)) * (x.shape[0] * width)]
    i = jnp.eye(x.shape[0] * width).repeat(x.shape[1] // (x.shape[0] * width), axis=-1)
    z = (x[:, None, :] * i[None, :, :]).sum(axis=-1)
    return z / (x.shape[1] // (x.shape[0] * width))

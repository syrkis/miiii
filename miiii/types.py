# %% types.py
#   miiii types
# by: Noah Syrkis

# Imports
from functools import cached_property
from chex import dataclass
from jaxtyping import Array
import jax.numpy as jnp


# %% Types
@dataclass
class Scope:
    train_acc: Array
    train_cce: Array
    valid_acc: Array
    valid_cce: Array


# %% Types
@dataclass
class Dataset:
    x: Array
    y: Array
    idxs: Array
    mask: Array  # mask away n-2 classes when doing binary classification
    task: Array  # correct for n-ary classification
    limit: Array
    primes: Array  # prime numbers used

    @property
    def train_x(self):
        return self.x[self.idxs][: self.limit]

    @property
    def valid_x(self):
        return self.x[self.idxs][self.limit :]

    @property
    def train_y(self):
        return self.y[self.idxs][: self.limit]

    @property
    def valid_y(self):
        return self.y[self.idxs][self.limit :]


@dataclass
class Params:
    tok: Array  # tok_emb
    pos: Array  # pos_emb
    out: Array
    i: Array
    o: Array
    k: Array
    q: Array
    v: Array
    p: Array


@dataclass
class State:
    params: Params
    opt_state: Params
    emas: Params

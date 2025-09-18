# %% types.py
#   miiii types

# Imports
from chex import dataclass
from dataclasses import field
from jaxtyping import Array
import jax.numpy as jnp


# %% Types
@dataclass
class Scope:
    train_acc: Array
    train_cce: Array
    valid_acc: Array
    valid_cce: Array
    neu: Array
    fft: Array


# %% Types
@dataclass
class Split:
    x: Array
    y: Array


@dataclass
class Dataset:
    idxs: Array
    mask: Array  # mask away n-2 classes when doing binary classification
    task: Array  # correct for n-ary classification
    primes: Array  # prime numbers used
    train: Split
    valid: Split
    x: Array = field(init=False)  # Exclude from __init__
    y: Array = field(init=False)  # Exclude from __init__

    def __post_init__(self):
        # Initialize arr as a jnp.array using x and y
        object.__setattr__(self, "x", jnp.concat((self.train.x, self.valid.x))[jnp.argsort(self.idxs)])
        object.__setattr__(self, "y", jnp.concat((self.train.y, self.valid.y))[jnp.argsort(self.idxs)])


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

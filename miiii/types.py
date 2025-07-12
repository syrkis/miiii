# Imports
from functools import cached_property
from chex import dataclass
from jaxtyping import Array

import jax.numpy as jnp


@dataclass
class Scope:
    acc: Array
    cce: Array


# %% Types
@dataclass
class Dataset:
    x: Array
    y: Array
    frac: Array
    idxs: Array
    mask: Array  # mask away n-2 classes when doing binary classification
    task: Array  # correct for n-ary classification
    primes: Array  # prime numbers used

    @property
    def t(self):
        return self.primes.size

    # @cached_property
    # def train(self):
    #     return self.x[self.idxs][: self.limit], self.y[self.idxs][: self.limit]

    # @cached_property
    # def valid(self):
    #     return self.x[self.idxs][self.limit :], self.y[self.idxs][self.limit :]

    # @cached_property
    # def limit(self):
    #     return jnp.int32(self.frac * self.x.shape[0])


@dataclass
class Params:
    tok: Array
    pos: Array
    out: Array  # should be a linear layer ?
    w_i: Array  # in
    w_o: Array  # out


@dataclass
class State:
    params: Params
    opt_state: Params
    emas: Params

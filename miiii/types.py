# Imports
from dataclasses import field
from functools import cached_property
from chex import dataclass
from jaxtyping import Array

import jax.numpy as jnp


# %% Types
@dataclass
class Dataset:
    x: Array
    y: Array
    idxs: Array
    mask: Array
    frac: float
    primes: Array
    weight: Array  # for masking tasks? Not sure

    @cached_property
    def train(self):
        return self.x[self.idxs][: self.limit], self.y[self.idxs][: self.limit]

    @cached_property
    def valid(self):
        return self.x[self.idxs][self.limit :], self.y[self.idxs][self.limit :]

    @cached_property
    def limit(self):
        return jnp.int32(self.frac * self.x.shape[0])


@dataclass
class Activation:
    wei: Array
    ffwd: Array = field(default_factory=lambda: jnp.array([]))
    logits: Array = field(default_factory=lambda: jnp.array([]))


@dataclass
class MetricSplit:
    loss: Array
    acc: Array


# %% Data classes
@dataclass
class Feedforward:
    w_i: Array  # in
    w_o: Array  # out


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array


@dataclass
class Params:
    embeds: Embedding
    ffwd: Feedforward
    unbeds: Array  # should be a linear layer ?


@dataclass
class Metrics:
    train: MetricSplit
    valid: MetricSplit


@dataclass
class State:
    params: Params
    opt_state: Params
    emas: Params

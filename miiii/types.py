# Imports
from dataclasses import field
from functools import cached_property
from chex import dataclass
from jaxtyping import Array
from typing import Tuple
import jax.numpy as jnp


# %% Types
@dataclass
class Dataset:
    x: Array
    y: Array
    idxs: Array
    fact_mask: Array  # for masking tasks? Not sure
    task_mask: Array
    primes: Array
    # size: Tuple[int, int] | Tuple[int, int, int]
    frac: float = 0.5

    @cached_property
    def train(self):
        return self.x[self.idxs][: self.limit], self.y[self.idxs][: self.limit]

    @cached_property
    def valid(self):
        return self.x[self.idxs][self.limit :], self.y[self.idxs][self.limit :]

    @cached_property
    def limit(self):
        return int(self.frac * self.x.shape[0])


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


# @dataclass
# class Attention:
# q: Array
# k: Array
# v: Array
# o: Array


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array


@dataclass
class Params:
    embeds: Embedding
    ffwd: Feedforward
    # attn: Attention
    unbeds: Array  # should be a linear layer ?


@dataclass
class Metrics:
    train: MetricSplit
    valid: MetricSplit


# @dataclass
# class Scope:
# logit_freqs: Array
# grad_norms: Params
# neuron_freqs: Array


@dataclass
class State:
    params: Params
    opt_state: Params
    emas: Params


@dataclass
class Conf:
    p: int = 113
    lamb: float = 2
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 20000
    lr: float = 3e-4  # i just usually do this.
    l2: float = 1.0
    dropout: float = 0.5
    train_frac: float = 0.5
    mask: bool = False  # weather to mask first four tasks
    shuffle: bool = False  # weather to shuffle the y labels
    alpha: float = 0.98

    @property
    def n(self):
        return self.p**2

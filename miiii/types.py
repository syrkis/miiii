# Imports
from dataclasses import field
from chex import dataclass
from jaxtyping import Array
import jax.numpy as jnp


# %% Types
@dataclass
class DataSplit:
    train: Array
    eval: Array


@dataclass
class Dataset:
    x: DataSplit
    y: DataSplit
    idxs: Array
    udxs: Array
    mask: Array
    primes: Array
    weight: Array | None = None


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
    w_in: Array
    w_out: Array


@dataclass
class Attention:
    q: Array
    k: Array
    v: Array
    o: Array


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array


@dataclass
class Params:
    embeds: Embedding
    ffwd: Feedforward
    attn: Attention
    unbeds: Array  # should be a linear layer ?


@dataclass
class Metrics:
    train: MetricSplit
    valid: MetricSplit


@dataclass
class Scope:
    # logit_freqs: Array
    grad_norms: Params | None
    neuron_freqs: Array


@dataclass
class State:
    params: Params
    opt_state: Params | None = None
    emas: Params | None = None


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

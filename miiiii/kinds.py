# types.py
#   miiii types and dataclasses
# by: Noah Syrkis

# %% Imports
from chex import dataclass
from typing import List
import optax
from jax import Array


@dataclass
class Metrics:
    train_loss: Array
    valid_loss: Array
    train_f1: Array
    valid_f1: Array


@dataclass
class Conf:
    # task is either "prime" or "prose"
    vocab_size: int
    batch_size: int
    seq_len: int
    task: str = "prime"  # "prose"
    causal: bool = False
    # initialixation scale for weights
    theta: float = 1e-2
    base: int = 2
    n: int = 1024
    latent_dim: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 100
    lr: float = 1e-3
    block: str = "vaswani"
    l2: float = 1e-4  # lambda
    dropout: float = 0.1


# %% Model classes
@dataclass
class Head:
    key: Array
    query: Array
    value: Array
    # proj: Array


@dataclass
class FFWD:
    w1: Array
    b1: Array
    w2: Array
    b2: Array


@dataclass
class Block:
    head: Head
    ffwd: FFWD


@dataclass
class Params:
    tok_emb: Array
    pos_emb: Array
    blocks: List[Block]
    lm_head: Array


# %% Train classes
@dataclass
class State:  # replace with chex and put in types
    params: Params
    opt_state: optax.OptState
    ema_grads: Array


# %% Data classes
@dataclass
class Datasplit:
    x: Array
    y: Array


@dataclass
class Datainfo:
    alpha: Array  # for a given tasks, the alpha probabilities of each class
    tasks: List[str]


@dataclass
class Dataset:
    train: Datasplit
    valid: Datasplit
    info: Datainfo
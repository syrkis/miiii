# types.py
#   miiii types and dataclasses
# by: Noah Syrkis

# %% Imports
from chex import dataclass
from typing import List
import optax
import jax.numpy as jnp
from jax import Array


# %% training dataclasses ##############################################################
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
    base: int = 2
    n: int = 1024
    latent_dim: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 100
    lr: float = 1e-3
    block: str = "vaswani"
    l2: float = 0.1
    dropout: float = 0.1


# %% model dataclasses ##############################################################
@dataclass
class Feedforward:
    w1: Array
    b1: Array
    w2: Array
    b2: Array


@dataclass
class Attention:
    q: Array
    k: Array
    v: Array
    p: Array


@dataclass
class Block:
    ffwd: Feedforward
    attn: Attention


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array


@dataclass
class Params:
    embeds: Embedding
    blocks: Block
    lm_out: Array  # should be a linear layer ?


# %% data dataclasses ##############################################################
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

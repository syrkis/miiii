# types.py
#   miiii types and dataclasses
# by: Noah Syrkis

# %% Imports
from chex import dataclass, Array
from typing import List


# %% Model classes
@dataclass
class Head:
    key: Array
    query: Array
    value: Array
    proj: Array


@dataclass
class FeedForward:
    w1: Array
    b1: Array
    w2: Array
    b2: Array


@dataclass
class Block:
    head: Head
    ffwd: FeedForward


@dataclass
class Params:
    tok_emb: Array
    pos_emb: Array
    blocks: List[Block]
    lm_head: Array


# %% Data classes
@dataclass
class Datasplit:
    x: Array
    y: Array


@dataclass
class Datainfo:
    apriori: Array  # for a given tasks, the apriori probabilities of each class
    tasks: List[str]


@dataclass
class Dataset:
    train: Datasplit
    valid: Datasplit
    info: Datainfo

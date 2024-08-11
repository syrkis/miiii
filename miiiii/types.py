# types.py
#   miiii types and dataclasses
# by: Noah Syrkis

# imports
from chex import dataclass
from typing import List, Set, Tuple


# dataclasses
@dataclass
class TrainState:
    params: dict
    opt_state: dict
    ema_grads: dict


@dataclass
class Config:
    base: int
    emb: int
    heads: int
    depth: int
    dropout: float
    n: int


@dataclass
class TrainConfig:
    base: int
    emb: int
    heads: int
    depth: int
    dropout: float
    n: int
    lr: float
    batch_size: int
    epochs: int
    warmup_steps: int
    decay_steps: int
    decay_rate: float
    alpha: float
    beta: float
    gamma: float
    epsilon: float
    seed: int
    device: str
    dtype: str
    data_dir: str
    model_dir: str
    log_dir: str
    save_dir: str


@dataclass
class DataConfig:
    base: int
    emb: int
    heads: int
    depth: int
    dropout: float
    n: int
    lr: float
    batch_size: int
    epochs: int
    warmup_steps: int
    decay_steps: int
    decay_rate: float
    alpha: float
    beta: float
    gamma: float
    epsilon: float
    seed: int
    device: str
    dtype: str
    data_dir: str
    model_dir: str
    log_dir: str
    save_dir: str

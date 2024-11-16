# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
from typing import Callable

import jax.numpy as jnp
import optax
from chex import dataclass
from jax import Array, random, vmap
from oeis import oeis
from typing import Tuple

from miiii.utils import Conf


# %% Data classes #################################################################
@dataclass
class Dataset:
    x_train: Array
    x_valid: Array
    y_train: Array
    y_valid: Array
    idxs: Array


@dataclass
class Task:
    type: str  # 'remainder', 'divisible'
    span: str  # 'atomic, 'batch'
    loss_fn: Callable


def task_fn(key: Array, cfg: Conf, task_type, task_span) -> Tuple[Dataset, Task]:
    match task_span:
        case "atomic":
            return nanda_fn(key, cfg, task_type, task_span)
        case "batch":
            return miiii_fn(key, cfg, task_type, task_span)
        case _:
            raise ValueError(f"task_span {task_span} not supported")


# miiii task  ################################################################
def repr_fn(x, p):  # represent x in base p (p is prime, and x is less than p**2)
    return jnp.stack((x // p, x % p, jnp.array(p).repeat(x.shape[0])), axis=-1).astype(jnp.int8)


### MULTI TRUE
def miiii_fn(key, cfg, task_type, task_span):
    primes = jnp.array(oeis["A000040"][1 : cfg.p])  # potential prime factors
    factors = primes[primes < cfg.p]  # prime factors
    idxs = random.permutation(key, jnp.arange(cfg.p**2))  # permute the indices
    x = repr_fn(jnp.arange(cfg.p**2), cfg.p)  # x is the representation of the numbers
    y = (jnp.arange(cfg.p**2)[:, None] % factors[None, :]).astype(jnp.int8)
    y = y if task_type == "remainder" else (y == 0).astype(jnp.int8)  # this was a serious bug i just fixed it
    x, y = x[idxs], y[idxs]
    x_train, y_train = x[: int(cfg.train_frac * cfg.p**2)], y[: int(cfg.train_frac * cfg.p**2)]
    x_valid, y_valid = x[int(cfg.train_frac * cfg.p**2) :], y[int(cfg.train_frac * cfg.p**2) :]
    loss = loss_fn(task_type, task_span)
    task = Task(loss_fn=loss, type=task_type, span=task_span)
    return Dataset(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, idxs=idxs), task


def loss_fn(task_type, task_span):
    match task_type, task_span:
        case "divisible", "atomic":  # focal_losos
            return focal_loss_fn
        case "remainder", "atomic":  # cross_entropy
            return cross_entropy_fn
        case "divisible", "batch":  # vmap focall loss
            return vmap(focal_loss_fn, in_axes=(1, 1, 0, None))
        case "remainder", "batch":  # vmap cross entropy
            return vmap(cross_entropy_fn, in_axes=(1, 1, None, None))
        case _:
            raise ValueError("Invalid task type or span")


# Train
def focal_loss_fn(logits, y, alpha, gamma):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    # consider squaring alpha, and increasing gamma?
    return optax.sigmoid_focal_loss(logits, y, alpha, gamma).astype(jnp.float32).mean()  # mean across samples


def cross_entropy_fn(logits, y, *_):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).astype(jnp.float32).mean()


# MULTI FALSE
# nanda task  ################################################################
def nanda_fn(key, cfg: Conf, task_type: str, task_span: str) -> Tuple[Dataset, Task]:
    # modular adition modulo prime
    a = jnp.arange(cfg.p).repeat(cfg.p)
    b = jnp.tile(jnp.arange(cfg.p), cfg.p)
    y = (a + b) % cfg.p
    data = jnp.stack([a, b, jnp.array(cfg.p).repeat(cfg.p**2), y], axis=-1)
    idxs = random.permutation(key, len(data))
    data = data[idxs]
    x = data[:, :-1]
    y = data[:, -1]
    x_train, x_valid = x[: int(len(x) * cfg.train_frac)], x[int(len(x) * cfg.train_frac) :]
    y_train, y_valid = y[: int(len(y) * cfg.train_frac)], y[int(len(y) * cfg.train_frac) :]
    if task_type == "divisible":
        y_train, y_valid = (y_train == 0).astype(jnp.int8), (y_valid == 0).astype(jnp.int8)
    loss = loss_fn(task_type, task_span)
    task = Task(loss_fn=loss, type=task_type, span=task_span)
    return Dataset(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid, idxs=idxs), task

# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
from typing import Callable

import jax.numpy as jnp
import optax
from chex import dataclass
from jax import Array, random, vmap, jit
from oeis import oeis
from typing import Tuple
from functools import partial

from miiii.utils import Conf


# %% Data classes #################################################################
@dataclass
class Split:
    train: Array
    eval: Array
    test: Array


@dataclass
class Dataset:
    x: Split
    y: Split
    # x_train: Array
    # x_eval: Array
    # x_test: Array
    # y_train: Array
    # y_eval: Array
    # y_test: Array
    idxs: Array


@dataclass
class Task:
    type: str  # 'remainder', 'divisible'
    span: str  # 'prime, 'factors'
    loss_fn: Callable
    mask: Array | None = None
    weight: Array | int = 1
    primes: Array | None = None


def task_fn(key: Array, cfg: Conf, task_type, task_span) -> Tuple[Dataset, Task]:
    match task_span:
        case "prime":
            return nanda_fn(key, cfg, task_type, task_span)
        case "factors":
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
    sep = int(cfg.train_frac * cfg.p**2)
    x_train, y_train = x[:sep], y[:sep]
    x_eval, y_eval = x[sep : sep + 1000], y[sep : sep + 1000]
    x_test, y_test = x[sep + 1000 :], y[sep + 1000 :]
    primes = jnp.array(oeis["A000040"][1 : y_train.shape[1] + 1])
    mask = jnp.tile(jnp.arange(primes.max()), primes.size).reshape((primes.size, -1)) < primes[:, None]
    mask = mask if task_type == "remainder" else jnp.array(1)
    loss = loss_fn(task_type, task_span, mask)
    # mask = jnp.zeros((*y_train.shape, primes.max())).at[:, mask].set(1).astype(jnp.bool)
    weight = jnp.log(
        mask.sum(-1)
    )  #  correct for number of classes in task. This is an good informational theoritical enhancement. Make it optional?
    task = Task(loss_fn=jit(loss), type=task_type, span=task_span, mask=mask, weight=weight, primes=primes)
    x = Split(train=x_train, eval=x_eval, test=x_test)
    y = Split(train=y_train, eval=y_eval, test=y_test)
    return Dataset(x=x, y=y, idxs=idxs), task


def loss_fn(task_type, task_span, mask):
    match task_type, task_span:
        case "divisible", "prime":  # focal_losos
            return focal_loss_fn
        case "remainder", "prime":  # cross_entropy
            return cross_entropy_fn
        case "divisible", "factors":  # vmap focall loss
            return vmap(focal_loss_fn, in_axes=(1, 1, 0, None, None))
        case "remainder", "factors":  # vmap cross entropy
            return vmap(cross_entropy_fn, in_axes=(1, 1, None, None, 0))
        case _:
            raise ValueError("Invalid task type or span")


# Train
def focal_loss_fn(logits, y, alpha, gamma, mask):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    # consider squaring alpha, and increasing gamma?
    return optax.sigmoid_focal_loss(logits, y, alpha, gamma).mean()  # mean across samples


def cross_entropy_fn(logits, y, alpha, gamma, mask):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    # print(mask.shape, logits.shape, y.shape)
    # exit()
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()


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
    sep = int(len(x) * cfg.train_frac)
    x_train, x_eval, x_test = x[:sep], x[sep : sep + 1000], x[sep + 1000 :]
    y_train, y_eval, y_test = y[:sep], y[sep : sep + 1000], y[sep + 1000 :]
    if task_type == "divisible":
        y_train, y_eval = (y_train == 0).astype(jnp.int8), (y_eval == 0).astype(jnp.int8)
    loss = loss_fn(task_type, task_span, mask=jnp.array(1))
    task = Task(loss_fn=jit(loss), type=task_type, span=task_span, mask=jnp.array(1))
    x = Split(train=x_train, eval=x_eval, test=x_test)
    y = Split(train=y_train, eval=y_eval, test=y_test)
    return Dataset(x=x, y=y, idxs=idxs), task

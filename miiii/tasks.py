# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
from miiii.utils import Conf
from oeis import oeis
import jax.numpy as jnp
from jax import Array, random
from typing import Tuple
from chex import dataclass


# %% Data classes #################################################################
# %% Functions
def task_fn(key: Array, cfg: Conf):
    return nanda_fn(key, cfg) if cfg.project == "nanda" else miiii_fn(key, cfg)


@dataclass
class Dataset:
    x_train: Array
    x_valid: Array
    y_train: Array
    y_valid: Array
    idxs: Array


# miiii task  ################################################################
def repr_fn(x, p):  # represent x in base p (p is prime, and x is less than p**2)
    return jnp.stack((x // p, x % p, jnp.array(p).repeat(x.shape[0])), axis=-1).astype(jnp.int8)


def miiii_fn(key, cfg):  # we go from 0 instead of 2 to avoid annoying index bugs
    primes = jnp.array(oeis["A000040"][1 : cfg.p])  # potential prime factors
    factors = primes[primes < cfg.p]  # prime factors
    idxs = random.permutation(key, jnp.arange(cfg.p**2))  # permute the indices
    x = repr_fn(jnp.arange(cfg.p**2), cfg.p)  # x is the representation of the numbers
    y = (jnp.arange(cfg.p**2)[:, None] % factors[None, :]).astype(jnp.int8)
    y = y if cfg.task == "multi" else (y == 0).astype(jnp.int8)
    x, y = x[idxs], y[idxs]
    x_train, y_train = x[: int(cfg.train_frac * cfg.p**2)], y[: int(cfg.train_frac * cfg.p**2)]
    x_valid, y_valid = x[int(cfg.train_frac * cfg.p**2) :], y[int(cfg.train_frac * cfg.p**2) :]
    return Dataset(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, idxs=idxs)


# nanda task  ################################################################
def nanda_fn(key, cfg: Conf) -> Dataset:
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
    return Dataset(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid, idxs=idxs)

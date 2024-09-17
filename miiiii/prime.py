# prime.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable, List, Tuple
from oeis import oeis


# %% Functions
def source_fn(n: int, base: int, ns: Callable) -> Array:
    x = ns(jnp.arange(n), base)
    return x


def target_fn(x: Array) -> Tuple[Array, List[str]]:
    all_primes = primes_fn(len(x))
    target_primes = all_primes[all_primes < len(x)]  # target primes
    test_primes = all_primes[all_primes < jnp.sqrt(len(x))]  # source primes
    is_prime = jnp.zeros(len(x)).at[target_primes].set(1).astype(jnp.int32)[:, None]
    is_multiple = (jnp.arange(len(x))[:, None] % test_primes == 0).astype(jnp.int32)
    y = jnp.concatenate([is_multiple, is_prime], axis=-1)
    tasks = list(map(str, test_primes.tolist())) + ["prime"]
    return y, tasks


def prime_fn(cfg: mi.kinds.Conf, key: Array | None = None) -> mi.kinds.Dataset:
    n, base = cfg.n, cfg.base
    x = source_fn(n, base, base_ns)  # get source
    y, tasks = target_fn(x)  # get target and tasks

    # shuffle data
    idxs = random.permutation(key, len(x)) if key is not None else jnp.arange(len(x))
    x, y = x[idxs], y[idxs]  # shuffle data

    sep = int(len(x) * 0.8)  # 80/20 split
    alpha = (1 - y[:sep].mean(axis=0)) ** 2  # for focal loss

    # dataset
    train = mi.kinds.Datasplit(x=x[:sep], y=y[:sep])
    valid = mi.kinds.Datasplit(x=x[sep:], y=y[sep:])
    info = mi.kinds.Datainfo(alpha=alpha, tasks=tasks)
    ds = mi.kinds.Dataset(train=train, valid=valid, info=info)

    return ds


def primes_fn(n: int) -> Array:
    limit = (n / jnp.log(n)).astype(jnp.int32)  # num primes less than n is n / ln(n)
    primes = jnp.array(oeis["A000040"][1 : limit * 2])  # get first limit primes
    assert max(primes) > n, "not enough primes"  # make sure there are enough primes
    primes = primes[primes < n]
    return primes


def base_ns(x, base):
    digits = digit_fn(x.max(), base)
    numb = jnp.array([x // base**i % base for i in range(digits)][::-1]).T
    return numb


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)

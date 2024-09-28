# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
from miiiii.utils import Conf, digit_fn
from oeis import oeis
import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable, List, Tuple
from chex import dataclass


# %% Data classes #################################################################
@dataclass
class Datasplit:
    x: Array
    y: Array


@dataclass
class Datainfo:
    alpha: Array  # for a given tasks, the alpha probabilities of each class
    tasks: List[str]
    idxs: Array  # idxs with which the dataset was shuffled


@dataclass
class Dataset:
    train: Datasplit
    valid: Datasplit
    info: Datainfo


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


def prime_fn(cfg: Conf, key: Array | None = None) -> Dataset:
    n, base = cfg.n, cfg.base
    x = source_fn(n, base, base_ns)  # get source
    y, tasks = target_fn(x)  # get target and tasks

    # shuffle data
    idxs = random.permutation(key, len(x)) if key is not None else jnp.arange(len(x))
    x, y = x[idxs], y[idxs]  # shuffle data

    sep = int(len(x) * 0.5)  # 50/50 split
    alpha = (1 - y[:sep].mean(axis=0)) ** 2  # for focal loss

    # dataset
    train = Datasplit(x=x[:sep], y=y[:sep])
    valid = Datasplit(x=x[sep:], y=y[sep:])
    info = Datainfo(alpha=alpha, tasks=tasks, idxs=idxs)
    ds = Dataset(train=train, valid=valid, info=info)

    return ds


def unsplit_fn(ds: Dataset) -> Datasplit:
    x = jnp.concat([ds.train.x, ds.valid.x], axis=0)[jnp.argsort(ds.info.idxs)]
    y = jnp.concat([ds.train.y, ds.valid.y], axis=0)[jnp.argsort(ds.info.idxs)]
    return Datasplit(x=x, y=y)


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


# nanda task  ################################################################
def nanda_fn():
    p = 113
    a, b = jnp.meshgrid(jnp.arange(p), jnp.arange(p))


# prose related to the tasks #############################################
# %% ## Ficciones function for testing
def prose_fn(rng, cfg, path: str = "data/ficciones.txt"):
    with open(path, "r") as f:
        text = f.read()

    c2i = {c: i for i, c in enumerate(sorted(list(set(text))))}
    data = encode_fn(text, c2i)
    train_data = data[: int(len(data) * 0.8)]
    eval_data = data[int(len(data) * 0.8) :]

    def batch_fn(rng, split):
        while True:
            rng, key = random.split(rng)
            length_limit = (len(split) - cfg.seq_len - 1) // cfg.batch_size * cfg.batch_size
            idxs = random.permutation(key, jnp.arange(length_limit))
            idxs = idxs[: len(idxs) - len(idxs) % cfg.batch_size].reshape(-1, cfg.batch_size)
            idxs = idxs[:, :, None] + jnp.arange(cfg.seq_len)
            for idx in idxs:
                x = split[idx]
                y = split[idx + 1]
                yield x, y

    i2c = {i: c for c, i in c2i.items()}
    cfg.vocab_size = len(c2i)
    rng, key = random.split(rng)
    return batch_fn(rng, train_data), batch_fn(key, eval_data), c2i, i2c, cfg


def encode_fn(text: str, c2i: dict) -> Array:
    return jnp.array([c2i[c] for c in text])


def decode_fn(x: Array, i2c: dict) -> str:
    return "".join([i2c[i] for i in x.tolist()])


if __name__ == "__main__":
    nanda_fn()

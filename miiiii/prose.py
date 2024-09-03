# data.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable, List, Tuple


# %% ## Ficciones function for testing
def prose_fn(rng, cfg):
    with open("data/ficciones.txt", "r") as f:
        text = f.read()

    c2i = {c: i for i, c in enumerate(sorted(list(set(text))))}
    data = encode_fn(text, c2i)

    def batch_fn(rng):
        while True:
            rng, key = random.split(rng)
            length_limit = (len(data) - cfg.seq_len - 1) // cfg.batch_size * cfg.batch_size
            idxs = random.permutation(key, jnp.arange(length_limit))
            idxs = idxs[: len(idxs) - len(idxs) % cfg.batch_size].reshape(-1, cfg.batch_size)
            idxs = idxs[:, :, None] + jnp.arange(cfg.seq_len)
            for idx in idxs:
                x = data[idx]
                y = data[idx + 1]
                yield x, y

    i2c = {i: c for c, i in c2i.items()}
    return batch_fn(rng), c2i, i2c


def encode_fn(text: str, c2i: dict) -> Array:
    return jnp.array([c2i[c] for c in text])


def decode_fn(x: Array, i2c: dict) -> str:
    return "".join([i2c[i] for i in x.tolist()])

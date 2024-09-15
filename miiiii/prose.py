# data.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable, List, Tuple


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

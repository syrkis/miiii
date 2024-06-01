# train.py
#   miiii train
# by: Noah Syrkis

# imports
import jax
from jax import random, grad, jit, value_and_grad
import optax
from tqdm import tqdm
import jax.numpy as jnp
from typing import List, Set, Tuple
from functools import partial
from oeis import oeis
from einops import rearrange


# functions
@jit
def loss_fn(params, x, y):  #  cross entropy loss
    logits = rearrange(apply_fn(params, x), "b t c -> (b t) c")
    labels = rearrange(y, "b t -> (b t)")
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()


@jit
def update_fn(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


@jit
def grad_fn(params, x, y):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    return loss, grads


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import apply_fn
    from utils import get_conf
    from datum import data_fn
    from numbs import base_n

    # x, y = data_fn(oeis["A000040"], conf["n_samples"], partial(base_n, n=conf["base"]))
    rng, key = random.split(random.PRNGKey(0))
    data, encode, decode, vocab = data_fn("ficciones", key, 16, 64)
    rng, key = random.split(rng)
    config = get_conf(in_d=len(vocab), out_d=len(vocab), len=next(data)[0].shape[1])
    params = init_fn(key, config)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    for i, (x, y) in zip(range(2500), data):
        loss, grads = grad_fn(params, x, y)
        params, opt_state = update_fn(params, grads, opt_state)
        print(i, loss)

    apply = jit(apply_fn)

    rng, key = random.split(rng)
    state = next(data)[0]
    for i in range(100):
        rng, key = random.split(rng)
        logits = apply(params, state[:, -8:])[:, -1, :]
        word = random.categorical(key, logits)[:, None]
        state = jnp.concatenate([state, word], axis=-1)

    print(decode(state[0].tolist()))

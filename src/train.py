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


# functions
@jit
def loss_fn(params, x, y):  #  bce
    pred = apply_fn(params, x)
    loss = -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))
    return loss


@jit
def update_fn(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def grad_fn(params, x, y):
    return value_and_grad(loss_fn)(params, x, y)


def est_loss(params, data):
    return jnp.mean(jnp.array([loss_fn(params, *next(data)) for _ in range(100)]))


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import apply_fn, generate_fn
    from utils import load_conf
    from datum import data_fn
    from numbs import base_n

    x, y = data_fn(oeis["A000040"], 2**10 - 2, partial(base_n, n=16))
    rng, key = random.split(random.PRNGKey(0))
    params = init_fn(key, load_conf(1))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    for i in range(1000):
        pred = apply_fn(params, x)
        loss, grads = grad_fn(params, x, y)
        params, opt_state = update_fn(params, grads, opt_state)
        print(loss.item())

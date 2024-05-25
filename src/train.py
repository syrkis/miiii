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


# functions
def loss_fn(params, x, y):
    pred = apply_fn(params, x)
    return jnp.mean((pred - y) ** 2)


@jit
def grad_fn(params, x, y):
    return value_and_grad(loss_fn)(params, x, y)


@jit
def update_fn(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import apply_fn
    from utils import load_conf
    from datum import conrad_fn

    data, c2i, i2c = conrad_fn(random.PRNGKey(0), 128)
    conf = load_conf(len(c2i))
    rng = random.PRNGKey(0)
    params = init_fn(rng, conf)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    for i in range(8):
        for j in tqdm(range(data.shape[1] - 1)):
            x, y = data[:, : j + 1], data[:, j + 1]
            pred = apply_fn(params, x)
            loss, grads = grad_fn(params, x, y)
            params, opt_state = update_fn(params, grads, opt_state)

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
@jit
def loss_fn(params, x, y):
    y = jax.nn.one_hot(y, 121)  # TODO: don't hardcode vocab size
    pred = apply_fn(params, x)
    loss = -jnp.mean(jnp.sum(pred * y, axis=-1))
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
    from datum import text_fn

    data, encode, decode, vocab = text_fn(random.PRNGKey(0), 32, 2)
    rng = random.PRNGKey(0)
    params = init_fn(rng, load_conf(len(vocab)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    for i in (pbar := tqdm(range(10000))):
        x, y = next(data)
        loss, grads = grad_fn(params, x, y)
        params, opt_state = update_fn(params, grads, opt_state)
        if i % (pbar.total // 10) == 0:
            pbar.set_description(f"loss: {est_loss(params, data):.3f}")
    x = generate_fn(params, x, rng)
    print(decode(x[0].tolist()))

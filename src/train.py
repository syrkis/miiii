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
import esch


# functions
def make_loss_fn(apply_fn):
    @jit
    def loss_fn(params, x, y):  #  cross entropy loss
        logits = apply_fn(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    return loss_fn


def make_update_fn(opt):
    @jit
    def update_fn(params, grads, opt_state):
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_fn


def make_grad_fn(loss_fn):
    @jit
    def grad_fn(params, x, y):
        loss, grads = value_and_grad(loss_fn, allow_int=True)(params, x, y)
        return loss, grads

    return grad_fn


def make_step_fn(grad_fn, update_fn, loss_fn, train_data, valid_data):
    @jit  # TODO: append losses during scan.
    def step_fn(carry, _=None):
        params, opt_state = carry
        train_loss, grads = grad_fn(params, *train_data)
        params, opt_state = update_fn(params, grads, opt_state)
        loss = jnp.array([train_loss, loss_fn(params, *valid_data)])
        return (params, opt_state), loss

    return step_fn


def train_fn(step_fn, params, opt_state, steps):
    state, losses = jax.lax.scan(step_fn, (params, opt_state), None, length=steps)
    return state, losses


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import get_conf
    from datum import data_fn
    from numbs import base_n

    seq = oeis["A000040"]  # "A000040" is the sequence of prime numbers
    data_conf, model_conf = get_conf()
    rng, key = random.split(random.PRNGKey(0))

    number_system = partial(base_n, data_conf["base"])
    train_data, valid_data = data_fn("primes", seq, data_conf["n"], number_system)

    params = init_fn(key, dict(**model_conf, len=train_data[0].shape[1]))

    apply_fn = make_apply_fn(vaswani_fn)
    loss_fn = make_loss_fn(apply_fn)
    grad_fn = make_grad_fn(loss_fn)

    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    update_fn = make_update_fn(opt)

    # train the model
    step_fn = make_step_fn(grad_fn, update_fn, loss_fn, train_data, valid_data)
    state, losses = train_fn(step_fn, params, opt_state, 1000)

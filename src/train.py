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
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        return loss, grads

    return grad_fn


def make_step_fn(grad_fn, update_fn, train_x, train_y):
    @jit  # TODO: append losses during scan.
    def step_fn(params, opt_state):
        loss, grads = grad_fn(params, train_x, train_y)
        params, opt_state = update_fn(params, grads, opt_state)
        return params, opt_state, loss

    return step_fn


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import load_conf
    from datum import prime_fn
    from numbs import base_n

    rng, key = random.split(random.PRNGKey(0))
    base, seq = 2, oeis["A000040"]
    number_system = partial(base_n, n=base)

    (train_x, train_y), (valid_x, valid_y) = prime_fn(seq, 2**14, number_system)
    config = dict(in_d=base, out_d=2, len=train_x.shape[1], **load_conf())

    params = init_fn(key, config)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # train functions
    apply_fn = make_apply_fn(vaswani_fn)
    loss_fn = make_loss_fn(apply_fn)
    grad_fn = make_grad_fn(loss_fn)

    update_fn = make_update_fn(opt)

    step_fn = make_step_fn(grad_fn, update_fn, train_x, train_y)
    state = jax.lax.scan(step_fn, params, init=opt_state, length=1000)

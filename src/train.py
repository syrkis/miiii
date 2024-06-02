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


def make_estimate_loss_fn(loss_fn):
    def estimate_loss(params, train_data, valid_data):
        train_loss = 0
        valid_loss = 0
        for i in range(10):
            train_loss += loss_fn(params, *next(train_data))
        for i in range(10):
            valid_loss += loss_fn(params, *next(valid_data))
        return train_loss / 10, valid_loss / 10

    return estimate_loss


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import load_conf
    from datum import data_fn
    from numbs import base_n

    base = 2
    x, y = data_fn("primes", oeis["A000040"], 2**10, partial(base_n, n=base))
    rng, key = random.split(random.PRNGKey(0))
    config = dict(in_d=base, out_d=2, len=x.shape[1], **load_conf())
    params = init_fn(key, config)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # train functions
    apply_fn = make_apply_fn(vaswani_fn)
    loss_fn = make_loss_fn(apply_fn)
    grad_fn = make_grad_fn(loss_fn)
    update_fn = make_update_fn(opt)

    # eval functions
    eval_apply_fn = jit(make_apply_fn(vaswani_fn))
    eval_loss_fn = make_loss_fn(eval_apply_fn)
    estimate_loss = make_estimate_loss_fn(eval_loss_fn)

    for i in range(10000):
        loss, grads = grad_fn(params, x, y)
        params, opt_state = update_fn(params, grads, opt_state)
        print(i, f"{loss:.4f}")

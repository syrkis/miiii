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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    from utils import alpha_fn
else:
    from .utils import alpha_fn


# functions
def make_loss_fn(apply_fn, conf, dropout=0.0):
    alpha = alpha_fn(conf.n // 2)

    @jit
    def loss_fn(params, rng, x, y):  #  cross entropy loss
        logits = apply_fn(params, rng, x, dropout)
        loss = optax.sigmoid_focal_loss(logits, y, alpha).mean()
        l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)) / 2
        return loss + conf.l2 * l2

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
    def grad_fn(params, rng, x, y):
        loss, grads = value_and_grad(loss_fn, allow_int=True)(params, rng, x, y)
        return loss, grads

    return grad_fn


def make_step_fn(grad_fn, update_fn, loss_fn, train_data, valid_data):
    @jit  # TODO: append losses during scan.
    def step_fn(carry, _=None):
        params, opt_state, rng = carry
        rng, key = random.split(rng)
        train_loss, grads = grad_fn(params, key, *train_data)
        params, opt_state = update_fn(params, grads, opt_state)
        loss = jnp.array([train_loss, loss_fn(params, rng, *valid_data)])  # dropout=0.0
        return (params, opt_state, rng), loss

    return step_fn


def make_train_fn(step_fn):
    def train_fn(steps, state):
        state, losses = jax.lax.scan(step_fn, state, None, length=steps)
        return state, losses

    return train_fn


def init_train(apply_fn, params, config, train_data, valid_data):
    opt = optax.adam(config.lr)
    train_loss_fn = make_loss_fn(apply_fn, config, dropout=config.dropout)
    valid_loss_fn = make_loss_fn(apply_fn, config, dropout=0.0)
    grad_fn = make_grad_fn(train_loss_fn)
    update_fn = make_update_fn(opt)
    opt_state = opt.init(params)
    step_fn = make_step_fn(grad_fn, update_fn, valid_loss_fn, train_data, valid_data)
    train_fn = make_train_fn(step_fn)
    return train_fn, opt_state


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import get_conf
    from datum import data_fn
    from numbs import base_n

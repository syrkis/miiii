# train.py
#   miiii train
# by: Noah Syrkis

# imports
import jax
from jax import random, grad, jit, value_and_grad
import optax
from jax import tree_util
from tqdm import tqdm
import jax.numpy as jnp
from typing import List, Set, Tuple
from functools import partial
from oeis import oeis
from einops import rearrange
import seaborn as sns
from typing import NamedTuple
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils import alpha_fn
else:
    from .utils import alpha_fn


# constants
class TrainState(NamedTuple):
    params: dict
    opt_state: dict
    ema_grads: dict


# functions
@jit
def loss_fn(logits, y):  #  cross entropy loss
    loss = optax.sigmoid_focal_loss(logits, y, 0.7).mean()
    return loss


def make_update_fn(opt):
    @jit
    def update_fn(params, grads, opt_state):
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_fn


def make_grad_fn(loss_fn, apply_fn, cfg):
    @jit
    def grad_fn(state, rng, x, y):  # maybe add allow_int flag below
        def loss_and_logits(params):
            logits = apply_fn(params, rng, x, cfg.dropout)
            loss = loss_fn(logits, y)
            return loss, logits

        (loss, logits), grads = value_and_grad(loss_and_logits, has_aux=True)(
            state.params
        )
        grads, state = gradfilter_ema(grads, state)  # @lee2024b (grokfast)
        return loss, grads, logits, state

    return grad_fn


@partial(jit, static_argnums=(2,))
def gradfilter_ema(grads, state, alpha=0.98, lamb=2.0):
    def _update_ema(prev_ema, grad):
        return prev_ema * alpha + grad * (1 - alpha)

    def _apply_ema(grad, ema):
        return grad + ema * lamb

    ema_grads = jax.tree_map(_update_ema, state.ema_grads, grads)
    filtered_grads = jax.tree_map(_apply_ema, grads, ema_grads)
    state = state._replace(ema_grads=ema_grads)
    return filtered_grads, state


def make_step_fn(grad_fn, update_fn, train_data, eval_fn):
    @jit
    def step_fn(state, rng):
        params, opt_state = state.params, state.opt_state
        rng, key = random.split(rng)
        loss, grads, logits, state = grad_fn(state, key, *train_data)
        params, opt_state = update_fn(params, grads, opt_state)
        metrics = eval_fn(params, rng, loss, logits)
        state = state._replace(params=params, opt_state=opt_state)
        return state, metrics

    return step_fn


def make_eval_fn(apply_fn, loss_fn, train_data, valid_data):
    @jit
    def eval_fn(params, rng, train_loss, train_logits):
        valid_logits = apply_fn(params, rng, valid_data[0], 0.0)
        valid_loss = loss_fn(valid_logits, valid_data[1])
        valid_pred = predict_fn(valid_logits)
        valid_acc = accuracy(valid_data[1], valid_pred)
        valid_f1 = f1_score(valid_data[1], valid_pred)

        train_pred = predict_fn(train_logits)
        train_acc = accuracy(train_data[1], train_pred)
        train_f1 = f1_score(train_data[1], train_pred)

        metrics = [train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1]
        return jnp.array(metrics)

    return eval_fn


def predict_fn(logits):
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


def f1_score(y_true, y_pred):
    tp = jnp.sum(y_true * y_pred)
    fp = jnp.sum((1 - y_true) * y_pred)
    fn = jnp.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def accuracy(y_true, y_pred):
    return jnp.mean(y_true == y_pred)


def make_train_fn(step_fn):
    def train_fn(steps, rng, state):
        rngs = random.split(rng, steps)
        state, metrics = jax.lax.scan(step_fn, state, rngs, length=steps)
        return state, metrics

    return train_fn


def init_train(apply_fn, params, cfg, train_data, valid_data):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=0.9, b2=0.98)  # @nanda2023
    opt_state = opt.init(params)

    update_fn = make_update_fn(opt)
    grad_fn = make_grad_fn(loss_fn, apply_fn, cfg)

    ema_grads = jax.tree_map(jnp.zeros_like, params)
    state = TrainState(params=params, opt_state=opt_state, ema_grads=ema_grads)

    eval_fn = make_eval_fn(apply_fn, loss_fn, train_data, valid_data)
    step_fn = make_step_fn(grad_fn, update_fn, train_data, eval_fn)

    train_fn = make_train_fn(step_fn)
    return train_fn, state


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import get_conf
    from datum import data_fn
    from numbs import base_n

# train.py
#   miiii train
# by: Noah Syrkis

# imports
import jax
from jax import random, grad, jit, value_and_grad
import optax
from jax import tree_util
from chex import dataclass
from tqdm import tqdm
import jax.numpy as jnp
from typing import List, Set, Tuple
from functools import partial
from oeis import oeis
from einops import rearrange
import seaborn as sns
from typing import NamedTuple
import matplotlib.pyplot as plt


# constants
class TrainState(NamedTuple):
    params: dict
    opt_state: dict
    ema_grads: dict


# functions
def make_loss_fn(cfg, alpha_fn):
    alpha = alpha_fn(cfg.n)

    @jit
    def loss_fn(logits, y):  #  cross entropy loss
        loss = optax.sigmoid_focal_loss(logits, y).mean()
        return loss

    return loss_fn


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
    # @lee2024b grokfast-like EMA gradient filtering
    def _update_ema(prev_ema, grad):
        return prev_ema * alpha + grad * (1 - alpha)

    def _apply_ema(grad, ema):
        return grad + ema * lamb

    ema_grads = jax.tree.map(_update_ema, state.ema_grads, grads)
    filtered_grads = jax.tree.map(_apply_ema, grads, ema_grads)
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
        valid_loss = loss_fn(valid_logits, valid_data[1])  # number

        train_metrics = metrics_fn(train_data[1], train_logits)
        valid_metrics = metrics_fn(valid_data[1], valid_logits)

        return train_loss, valid_loss, train_metrics, valid_metrics

    return eval_fn


def predict_fn(logits):
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


def metrics_fn(y_true, logits):
    """returns accuracy, precision, recall, f1"""
    y_pred = predict_fn(logits)
    tp = jnp.sum(y_true * y_pred, axis=0)
    fp = jnp.sum((1 - y_true) * y_pred, axis=0)
    fn = jnp.sum(y_true * (1 - y_pred), axis=0)
    acc = (y_true == y_pred).mean(axis=0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    loss = optax.sigmoid_focal_loss(logits, y_true).mean(axis=0)
    return loss, f1, prec, rec, acc


def make_train_fn(step_fn):
    def train_fn(steps, rng, state):
        rngs = random.split(rng, steps)
        state, metrics = jax.lax.scan(step_fn, state, rngs, length=steps)
        return state, metrics

    return train_fn


def init_train(apply_fn, params, cfg, alpha_fn, train_data, valid_data):
    loss_fn = make_loss_fn(cfg, alpha_fn)
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=0.9, b2=0.98)  # @nanda2023
    opt_state = opt.init(params)

    update_fn = make_update_fn(opt)
    grad_fn = make_grad_fn(loss_fn, apply_fn, cfg)

    ema_grads = jax.tree.map(jnp.zeros_like, params)
    state = TrainState(params=params, opt_state=opt_state, ema_grads=ema_grads)

    eval_fn = make_eval_fn(apply_fn, loss_fn, train_data, valid_data)
    step_fn = make_step_fn(grad_fn, update_fn, train_data, eval_fn)

    train_fn = make_train_fn(step_fn)
    return train_fn, state


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import get_conf, alpha_fn, digit_fn
    from datum import prime_fn
    from numbs import base_ns

    seed = 0
    cfg = get_conf()
    rng, key = random.split(random.PRNGKey(seed))
    ns = partial(base_ns, digit_fn)
    (x_train, y_train), (x_valid, y_valid) = prime_fn(cfg.n, cfg.base, ns, key)
    params = init_fn(rng, cfg, x_train, y_train)

    apply_fn = make_apply_fn(vaswani_fn)
    train_fn, state = init_train(
        apply_fn, params, cfg, alpha_fn, (x_train, y_train), (x_valid, y_valid)
    )
    state, metrics = train_fn(cfg.epochs, rng, state)

    print(metrics)

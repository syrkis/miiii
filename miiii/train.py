# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from typing import Tuple

import jax.numpy as jnp
import optax
from jax import lax, random, tree, value_and_grad, vmap, jit
from jax_tqdm import scan_tqdm
from functools import partial
from typing import cast
from jaxtyping import Array

from miiii.model import apply_fn, init_fn
from miiii.tasks import Dataset
from miiii.utils import Conf, Metrics, Params, Split, State


ADAM_BETA1 = 0.9  # @nanda2023
ADAM_BETA2 = 0.98  # @nanda2023
ALPHA = 0.98


# %% Functions
def make_update_fn(opt, grad_fn, ds, cfg, arg):
    @jit
    def update_fn(state, key):
        loss, grads = grad_fn(state.params, key)
        grads, emas = filter_fn(grads, state.emas, cfg.lamb)
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = cast(Params, optax.apply_updates(state.params, updates))
        state = State(params=params, emas=emas, opt_state=opt_state)
        return state, loss

    return update_fn


def make_grad_fn(ds: Dataset, cfg: Conf, arg, apply, loss_fn):
    mask, weight = ds.mask, ds.weight

    @value_and_grad
    def grad_fn(params: Params, rng) -> Array:
        logits = apply(rng, params, ds.x.train)
        losses = loss_fn(logits, ds.y.train, mask) * weight
        return losses.mean()

    return grad_fn


@jit
def filter_fn(grads, emas, lamb: float):
    emas = tree.map(lambda grad, ema: ema * ALPHA + grad * (1 - ALPHA), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def make_interval_fn(cfg, arg, opt, loss_fn, ds):
    train_apply = apply_fn(cfg, ds, dropout=cfg.dropout)
    grad_fn = make_grad_fn(ds, cfg, arg, train_apply, loss_fn)
    update_fn = make_update_fn(opt, grad_fn, ds, cfg, arg)

    @scan_tqdm(arg.tick)
    def interval_fn(state, inputs):
        epoch, rng = inputs
        keys = random.split(rng, cfg.epochs // arg.tick)
        state, loss = lax.scan(update_fn, state, keys)
        return state, loss

    return interval_fn


def init_state(rng, cfg: Conf, arg, ds: Dataset):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)  # @nanda2023
    params: Params = init_fn(rng, cfg, arg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = cast(Params, opt.init(params))  # type: ignore
    return State(params=params, opt_state=opt_state, emas=emas), opt


def train_fn(rng, cfg: Conf, arg, ds: Dataset) -> Tuple[State, Array]:
    loss_fn = vmap(cross_entropy, in_axes=(1, 1, 0))
    state, opt = init_state(rng, cfg, arg, ds)
    interval_fn = make_interval_fn(cfg, arg, opt, loss_fn, ds)

    inputs = (jnp.arange(arg.tick), random.split(rng, arg.tick))
    state, loss = lax.scan(interval_fn, state, inputs)
    return state, loss


# def focal_loss_fn(logits, y, alpha, gamma, mask):
# logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
# return optax.sigmoid_focal_loss(logits, y, alpha, gamma).mean()  # mean across samples


def cross_entropy(logits, y, mask):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()


# Evaluate
def make_eval_fn(ds: Dataset, cfg: Conf, arg, apply, loss_fn):
    apply = partial(apply, random.PRNGKey(0))

    def acc_fn(y_pred, y_true):
        return (y_pred == y_true).mean()

    acc_fn = vmap(acc_fn, in_axes=(1, 1)) if arg.task == "miiii" else acc_fn
    loss_fn, mask, weight = jit(loss_fn), ds.mask, ds.weight

    def aux(params, x, y):
        logits = apply(params, x)
        losses = loss_fn(logits, y, mask) * weight
        accuracy = acc_fn(logits.argmax(-1), y)
        return Split(loss=losses, acc=accuracy)

    @jit
    def evaluate(state: State):
        valid_metrics = aux(state.params, ds.x.eval, ds.y.eval)
        train_metrics = aux(state.params, ds.x.train, ds.y.train)
        metrics = Metrics(train=train_metrics, valid=valid_metrics)
        return metrics

    return evaluate

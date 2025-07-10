# %% train.py
#   miiii train
# by: Noah Syrkis

# Imports
from functools import partial
from typing import cast

import jax.numpy as jnp
import optax
from jax import lax, random, tree, value_and_grad, vmap
from typing import Tuple
from optax import GradientTransformation
from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii.model import apply, init_fn as _init_fn
from miiii.tasks import Dataset
from miiii.types import Conf, Metrics, Params, MetricSplit, State


# Constants
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98


# %% Functions
def train_fn(rng, cfg: Conf, arg, ds: Dataset, state: State, opt: GradientTransformation):
    update = partial(update_fn, opt, ds, cfg, arg)
    state, loss = update(state, rng)

    @scan_tqdm(arg.tick)
    def step(state, idx_rng):
        keys = random.split(idx_rng[1], cfg.epochs // arg.tick)
        state, loss = lax.scan(update, state, keys)
        return state, loss

    input = (jnp.arange(arg.tick), random.split(rng, arg.tick))
    state, loss = lax.scan(step, state, input)

    return state, loss


def update_fn(opt: GradientTransformation, ds: Dataset, cfg: Conf, arg, state: State, key) -> Tuple[State, Array]:
    loss, grad = value_and_grad(partial(loss_fn, ds, arg))(state.params, key)
    grad, emas = filter_fn(grad, state.emas, cfg.lamb, cfg.alpha)
    updates, opt_state = opt.update(grad, state.opt_state, state.params)  # type: ignore
    params = optax.apply_updates(state.params, updates)  # type: ignore
    return State(params=params, emas=emas, opt_state=opt_state), loss  # type: ignore


def loss_fn(ds: Dataset, arg, params: Params, rng) -> Array:
    logits, z = apply(ds, rng, params, ds.x)
    losses = cross_entropy(logits, ds.y, ds.task_mask) * ds.task_mask
    return losses.mean()


def filter_fn(grad: Params, emas: Params, lamb: float, alpha: float) -> Tuple[Params, Params]:
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grad, emas)
    grad = tree.map(lambda grad, ema: grad + lamb * ema, grad, emas)
    return grad, emas


# %% INIT
def init_fn(rng, cfg: Conf, arg, ds: Dataset) -> Tuple[State, GradientTransformation]:
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)
    params = _init_fn(rng, cfg, arg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = cast(Params, opt.init(params))  # type: ignore
    return State(params=params, opt_state=opt_state, emas=emas), opt


# %% HELPERS
@partial(vmap, in_axes=(1, 1, 0))
def cross_entropy(logits: Array, y: Array, mask: Array):
    logits = logits.astype(jnp.float64)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()


def make_eval_fn(ds: Dataset, cfg: Conf, arg, loss_fn):
    metrics_fn = make_metrics_fn(apply, loss_fn, arg, ds)

    def eval_fn(state: State):
        valid_metrics = metrics_fn(state.params, ds.valid[0], ds.valid[1])
        train_metrics = metrics_fn(state.params, ds.train[0], ds.train[1])
        metrics = Metrics(train=train_metrics, valid=valid_metrics)
        return metrics

    return eval_fn


def make_acc_fn(arg):
    def acc_fn(y_pred, y_true):
        return (y_pred == y_true).mean()

    acc_fn = vmap(acc_fn, in_axes=(1, 1)) if arg.task == "miiii" else acc_fn
    return acc_fn


def make_metrics_fn(apply_fn, loss_fn, arg, ds):
    acc_fn = make_acc_fn(arg)

    def metrics_fn(params, x, y):
        logits, _ = apply_fn(params, x)
        losses = loss_fn(logits, y, ds.mask) * ds.weight
        accuracy = acc_fn(logits.argmax(-1), y)
        return MetricSplit(loss=losses, acc=accuracy)

    return metrics_fn

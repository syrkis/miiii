# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from typing import Tuple

import jax.numpy as jnp
import optax
from jax import lax, random, tree, value_and_grad, vmap
from jax_tqdm import scan_tqdm
from functools import partial
from typing import cast
from jaxtyping import Array

from miiii.model import apply_fn, init_fn
from miiii.tasks import Dataset, Task
from miiii.utils import Conf, Metrics, Params, Split, State


ADAM_BETA1 = 0.9  # @nanda2023
ADAM_BETA2 = 0.98  # @nanda2023
ALPHA = 0.98


# %% Functions
def update_fn(opt, ds: Dataset, task: Task, cfg: Conf):
    train_apply = apply_fn(cfg, ds, task, eval=False)
    valid_apply = partial(apply_fn(cfg, ds, task, eval=True), random.PRNGKey(0))
    grad = grad_fn(ds, task, cfg, train_apply)

    def update(state, key):
        loss, grads = grad(state.params, key)
        grads, emas = filter_fn(grads, state.emas, cfg.lamb)  # grokfast values
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = cast(Params, optax.apply_updates(state.params, updates))
        state = State(params=params, emas=emas, opt_state=opt_state)
        return state, loss

    return update, train_apply, valid_apply


def grad_fn(ds: Dataset, task: Task, cfg: Conf, apply):
    loss_fn, mask, weight = task.loss_fn, task.mask, task.weight

    @value_and_grad
    def grad(params: Params, rng) -> Array:
        acts = apply(rng, params, ds.x.train)
        losses = loss_fn(acts.logits, ds.y.train, 1 - ds.y.train.mean(0), 2, mask) * weight
        return losses.mean()

    return grad


def filter_fn(grads, emas, lamb: float):
    emas = tree.map(lambda grad, ema: ema * ALPHA + grad * (1 - ALPHA), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds: Dataset, task: Task, cfg: Conf, opt, scope):
    update, train_apply, valid_apply = update_fn(opt, ds, task, cfg)
    evaluate = evaluate_fn(ds, task, cfg, valid_apply)

    @scan_tqdm(100)
    def step(state, args):
        epoch, key = args
        keys = random.split(key, cfg.epochs // 100)
        state, loss = lax.scan(update, state, keys)
        return state, evaluate(state)

    return step


def init_state(rng, cfg: Conf, ds: Dataset, task: Task, opt):
    params: Params = init_fn(rng, cfg, ds, task)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)
    return State(params=params, opt_state=opt_state, emas=emas)


def train(rng, cfg: Conf, ds: Dataset, task: Task, scope=False) -> Tuple[State, Metrics]:
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)  # @nanda2023
    state = init_state(rng, cfg, ds, task, opt)
    step = step_fn(ds, task, cfg, opt, scope)
    state, metrics = lax.scan(step, state, (jnp.arange(100), random.split(rng, 100)))
    return state, metrics


# Evaluate
def evaluate_fn(ds: Dataset, task: Task, cfg: Conf, apply):
    def acc_fn(y_pred, y_true):
        return (y_pred == y_true).mean()

    acc_fn = vmap(acc_fn, in_axes=(1, 1)) if task.span == "factors" else acc_fn
    loss_fn, mask, weight = task.loss_fn, task.mask, task.weight

    def aux(state: State, x, y):
        acts = apply(state.params, x)
        losses = loss_fn(acts.logits, y, 1 - y.mean(0), 2, mask) * weight
        accuracy = acc_fn(acts.logits.argmax(-1), y)
        return Split(loss=losses, acc=accuracy)

    def evaluate(state: State):
        valid_metrics = aux(state, ds.x.eval, ds.y.eval)
        train_metrics = aux(state, ds.x.train, ds.y.train)
        metrics = Metrics(train=train_metrics, valid=valid_metrics)
        return metrics

    return evaluate

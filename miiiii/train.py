# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
# import miiiii as mi
from miiiii.model import Params, apply_fn, init_fn
from miiiii.tasks import Dataset
from miiiii.utils import Conf

from jax import random, jit, value_and_grad, vmap, nn, lax, tree
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import optax
from functools import partial
from typing import Tuple
from chex import dataclass, Array


@dataclass
class State:
    params: Array
    opt_state: Array
    emas: Array


@dataclass
class Metrics:
    train_loss: Array
    valid_loss: Array
    train_f1: Array
    valid_f1: Array


# functions
@partial(vmap, in_axes=(1, 1, 0))  # vmap across task (not sample)
def loss_fn(logits, y, alpha):
    # logits = logits.astype(jnp.float64)
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha).mean()  # do not take mean (task loss vector)


def update_fn(opt, ds, cfg):
    apply = apply_fn(cfg)

    def update(params, opt_state, emas, key):
        loss, losses, logits, grads = grad_fn(params, key, ds, cfg, apply)
        grads, emas = filter_fn(grads, emas, 0.98, 2)  # grokfast values
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, emas), (loss, losses, logits)

    return update, apply


def grad_fn(
    params: Params, rng: jnp.ndarray, ds: Dataset, cfg: Conf, apply
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def loss_and_logits(params: Params) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        logits = apply(params, rng, ds.train.x, cfg.dropout)
        losses = loss_fn(logits, ds.train.y, ds.info.alpha)  # mean for optimization
        return losses.mean(), (losses, logits)

    (loss, (losses, logits)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return loss, losses, logits, grads


def filter_fn(grads, emas, alpha: float, lamb: float):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds, cfg, opt):
    evaluate = evaluate_fn(ds, cfg)
    update, apply = update_fn(opt, ds, cfg)

    @scan_tqdm(cfg.epochs)
    def step(state, args):
        (params, opt_state, emas), (epoch, key) = state, args
        (params, opt_state, emas), (loss, losses, logits) = update(params, opt_state, emas, key)
        metrics = evaluate(params, key, logits, losses, apply)
        return (params, opt_state, emas), metrics

    return step


def init_state(rng, cfg, opt):
    params = init_fn(rng, cfg)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)  # type: ignore
    return params, opt_state, emas


def train(rng, cfg, ds):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=0.9, b2=0.98)
    state = init_state(rng, cfg, opt)
    step = step_fn(ds, cfg, opt)
    rngs = random.split(rng, cfg.epochs)
    state, metrics = lax.scan(step, state, (jnp.arange(cfg.epochs), rngs))
    return state, metrics


# Evaluation #########################################################################
def evaluate_fn(ds, cfg):
    def evaluate(params, key, train_logits, train_losses, apply):
        valid_logits = apply(params, key, ds.valid.x, cfg.dropout)
        valid_losses = loss_fn(valid_logits, ds.valid.y, ds.info.alpha)
        train_f1 = f1_score(train_logits, ds.train.y)
        valid_f1 = f1_score(valid_logits, ds.valid.y)
        return Metrics(train_f1=train_f1, valid_f1=valid_f1, train_loss=train_losses, valid_loss=valid_losses)

    return evaluate


@partial(vmap, in_axes=(1, 1))
def f1_score(y_logits, y_true):
    y_pred = (nn.sigmoid(y_logits) > 0.5).astype(jnp.int32)  # 0.5 threshold
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

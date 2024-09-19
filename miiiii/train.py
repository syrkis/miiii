# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax
from jax import random, jit, value_and_grad, vmap, nn, lax, tree
import optax
import jax.numpy as jnp
from functools import partial
from aim import Run, Figure
from jax_tqdm import scan_tqdm
from tqdm import tqdm
import numpy as np


# functions
@partial(vmap, in_axes=(1, 1, 0))  # vmap across task (not sample)
def loss_fn(logits, y, alpha):
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha).mean()  # do not take mean (task loss vector)


def update_fn(opt, ds, cfg):
    def update(params, opt_state, emas, key):
        losses, logits, grads = mi.train.grad_fn(params, key, ds, cfg)
        grads, emas = filter_fn(grads, emas, 0.98, 2)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, emas, losses, logits

    return update


def grad_fn(params, rng, ds, cfg):  # maybe add allow_int flag below
    def loss_and_logits(params):
        logits = mi.model.apply(params, rng, ds.train.x, cfg.dropout)
        losses = loss_fn(logits, ds.train.y, ds.info.alpha)  # mean for optimization
        return losses.mean(), (losses, logits)

    (_, (losses, logits)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return losses, logits, grads


def filter_fn(grads, emas, alpha, lamb):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds, cfg, opt):
    evaluate = evaluate_fn(ds, cfg)
    update = update_fn(opt, ds, cfg)

    @jit
    def step(state, key):
        params, opt_state, emas = state
        params, opt_state, emas, losses, logits = update(params, opt_state, emas, key)
        metrics = evaluate(params, key, logits, losses)
        return (params, opt_state, emas), metrics

    return step


def train(rng, cfg, ds):
    params = mi.model.init_fn(rng, cfg)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt = optax.adamw(cfg.lr, b1=0.9, b2=0.98, weight_decay=cfg.l2)
    opt_state = opt.init(params)  # type: ignore
    step = step_fn(ds, cfg, opt)
    rngs = random.split(rng, cfg.epochs)
    state = (params, opt_state, emas)

    # state, metrics = lax.scan(step, state, rngs)  # sometimes getting GPU metal shader error
    # return lax.scan(step, state, rngs)  # sometimes getting GPU metal shader error

    #########################################
    metrics = []
    for key in (pbar := tqdm(rngs)):
        state, metric = step(state, key)
        metrics.append(metric)

    metrics = {
        "train": {
            "f1": np.array([m.train_f1 for m in metrics]).T,
            "loss": np.array([m.train_loss for m in metrics]).T,
        },
        "valid": {
            "f1": np.array([m.valid_f1 for m in metrics]).T,
            "loss": np.array([m.valid_loss for m in metrics]).T,
        },
    }
    #########################################
    return state, metrics


# Evaluation #########################################################################
def evaluate_fn(ds, cfg):
    def evaluate(params, key, train_logits, train_losses):
        valid_logits = mi.model.apply(params, key, ds.valid.x, cfg.dropout)
        valid_losses = mi.train.loss_fn(valid_logits, ds.valid.y, ds.info.alpha)
        train_f1 = f1_score(train_logits, ds.train.y)
        valid_f1 = f1_score(valid_logits, ds.valid.y)
        return mi.kinds.Metrics(train_f1=train_f1, valid_f1=valid_f1, train_loss=train_losses, valid_loss=valid_losses)

    return evaluate


@partial(vmap, in_axes=(1, 1))
def f1_score(y_logits, y_true):
    y_pred = (nn.sigmoid(y_logits) > 0.5).astype(jnp.int32)  # 0.5 threshold
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

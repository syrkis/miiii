# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from miiiii.model import Params, apply_fn, init_fn, Output
from miiiii.tasks import Dataset
from miiiii.utils import Conf

from jax import random, jit, value_and_grad, vmap, nn, lax, tree
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import optax
from functools import partial
from typing import Tuple
from chex import dataclass, Array
import wandb


@dataclass
class State:
    params: Params | optax.Params
    opt_state: optax.OptState
    emas: Params


@dataclass
class Split:
    loss: Array
    f1: Array
    acc: Array


@dataclass
class Metrics:
    train: Split
    valid: Split


# functions
@partial(vmap, in_axes=(1, 1, 0))  # type: ignore vmap across task (not sample)
def focal_loss_fn(logits, y, alpha):
    # logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha).mean()  # mean across samples


def cross_entropy_loss_fn(logits, y, _):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def update_fn(opt, ds, cfg: Conf):
    apply = apply_fn(cfg)
    loss_fn = focal_loss_fn if cfg.task == "miiii" else cross_entropy_loss_fn

    def update(state, key):
        loss, losses, output, grads = grad_fn(state.params, key, ds, cfg, apply, loss_fn)
        # grads, emas = filter_fn(grads, state.emas, 0.98, 2)  # grokfast values
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = State(params=params, emas=state.emas, opt_state=opt_state)
        return state, (loss, losses, output)

    return update, apply, loss_fn


def grad_fn(params: Params, rng, ds: Dataset, cfg: Conf, apply, loss_fn) -> Tuple[Array, Array, Output, Array]:
    def loss_and_logits(params: Params) -> Tuple[jnp.ndarray, Tuple[Array, Output]]:
        output: Output = apply(params, rng, ds.train.x, cfg.hyper.dropout)
        losses = loss_fn(output.logits, ds.train.y, ds.info.alpha)  # mean for optimization
        return losses.mean(), (losses, output)

    (loss, (losses, output)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return loss, losses, output, grads


def filter_fn(grads, emas, alpha: float, lamb: float):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds, cfg: Conf, opt, scope):  # scope is for showing activations during trainig
    opt = optax.adamw(cfg.hyper.lr, weight_decay=cfg.hyper.l2, b1=0.9, b2=0.98)
    update, apply, loss_fn = update_fn(opt, ds, cfg)
    evaluate = evaluate_fn(ds, cfg, apply, loss_fn)

    @scan_tqdm(cfg.hyper.epochs)
    def step(state, args):
        epoch, key = args
        state, (loss, losses, output) = update(state, key)
        metrics = evaluate(state.params, key, losses, output.logits)
        return state, (metrics, output if scope else None)

    return step


def init_state(rng, cfg: Conf, opt):
    params = init_fn(rng, cfg)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)
    return State(params=params, opt_state=opt_state, emas=emas)


def train(rng, cfg: Conf, ds, scope=False, log=False):
    opt = optax.adamw(cfg.hyper.lr, weight_decay=cfg.hyper.l2, b1=0.9, b2=0.98)
    state = init_state(rng, cfg, opt)
    step = step_fn(ds, cfg, opt, scope)
    rngs = random.split(rng, cfg.hyper.epochs)
    state, (metrics, outputs) = lax.scan(step, state, (jnp.arange(cfg.hyper.epochs), rngs))
    log_fn(cfg, ds, metrics) if log else None
    return state, metrics, outputs


def log_fn(cfg: Conf, ds: Dataset, metrics: Metrics):
    hyper = cfg.hyper
    cfg.__dict__.__delitem__("hyper")
    config = cfg.__dict__ | hyper.__dict__
    wandb.init(project=cfg.task, config=config, entity="syrkis", mode="offline")
    for epoch in range(hyper.epochs):
        wandb.log(
            {
                "train_loss": metrics.train.loss[epoch].item(),  # type: ignore
                "train_f1": metrics.train.f1[epoch].item(),  # type: ignore
                "train_acc": metrics.train.acc[epoch].item(),  # type: ignore
                "valid_loss": metrics.valid.loss[epoch].item(),  # type: ignore
                "valid_f1": metrics.valid.f1[epoch].item(),  # type: ignore
                "valid_acc": metrics.valid.acc[epoch].item(),  # type: ignore
            },
            step=epoch,
        )

    # sync wandb
    wandb.finish()


# Evaluation #########################################################################
def evaluate_fn(ds, cfg: Conf, apply, loss_fn):
    f1_score = vmap(f1_score_fn, in_axes=(1, 1)) if cfg.task == "miiii" else f1_score_fn  # type: ignore
    accuracy = vmap(accuracy_fn, in_axes=(1, 1)) if cfg.task == "miiii" else accuracy_fn  # type: ignore
    pred_fn = (lambda l: (nn.sigmoid(l) > 0.5).astype(int)) if cfg.task == "miiii" else lambda l: jnp.argmax(l, axis=-1)

    def aux_fn(logits, y, loss):
        pred = pred_fn(logits)
        f1, acc = f1_score(pred, y), accuracy(pred, y)
        return Split(loss=loss, f1=f1, acc=acc)

    def evaluate(params, key, train_loss, train_logits):
        valid_output = apply(params, key, ds.valid.x, cfg.hyper.dropout)
        valid_loss = loss_fn(valid_output.logits, ds.valid.y, ds.info.alpha)

        valid_metrics = aux_fn(valid_output.logits, ds.valid.y, valid_loss)
        train_metrics = aux_fn(train_logits, ds.train.y, train_loss)

        return Metrics(train=train_metrics, valid=valid_metrics)

    return evaluate


def accuracy_fn(y_pred, y_true):
    y_pred, y_true = y_pred.flatten().astype(int), y_true.flatten().astype(int)
    return (y_pred == y_true).mean()


def f1_score_fn(y_pred, y_true):
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

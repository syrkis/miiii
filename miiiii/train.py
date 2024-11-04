# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from miiiii.model import Params, apply_fn, init_fn, Activation
from miiiii.tasks import Dataset
from miiiii.utils import Conf, log_fn

from jax import random, jit, value_and_grad, vmap, nn, lax, tree
from jax.experimental import io_callback
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
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha).mean()  # mean across samples


def cross_entropy_loss_fn(logits, y, _):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def update_fn(opt, ds, cfg: Conf):
    apply = apply_fn(cfg)
    loss_fn = focal_loss_fn if cfg.project == "miiii" else cross_entropy_loss_fn

    def update(state, key):
        loss, losses, output, grads = grad_fn(state.params, key, ds, cfg, apply, loss_fn)
        grads, emas = filter_fn(grads, state.emas, cfg.alpha, cfg.lamb)  # grokfast values
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = State(params=params, emas=emas, opt_state=opt_state)
        return state, (loss, losses, output)

    return update, apply, loss_fn


def grad_fn(params: Params, rng, ds: Dataset, cfg: Conf, apply, loss_fn) -> Tuple[Array, Array, Activation, Array]:
    def loss_and_logits(params: Params) -> Tuple[jnp.ndarray, Tuple[Array, Activation]]:
        output: Activation = apply(params, rng, ds.train.x, cfg.dropout)
        losses = loss_fn(output.logits, ds.train.y, ds.info.alpha)  # mean for optimization
        # print(losses.shape, ds.info.task.shape)
        # exit()
        return (losses * ds.info.task).sum(), (losses, output)

    (loss, (losses, output)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return loss, losses, output, grads


def filter_fn(grads, emas, alpha: float, lamb: float):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds, cfg: Conf, opt, scope):  # scope is for showing activations during trainig
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2)  # b1=0.9, b2=0.98)
    update, apply, loss_fn = update_fn(opt, ds, cfg)
    evaluate = evaluate_fn(ds, cfg, apply, loss_fn)

    @scan_tqdm(cfg.epochs)
    def step(state, args):
        epoch, key = args
        state, (loss, losses, train_output) = update(state, key)
        metrics, valid_output = evaluate(state.params, key, losses, train_output.logits)
        output = tree.map(lambda a, b: jnp.concat((a, b))[ds.info.udxs].astype(jnp.float16), train_output, valid_output)
        return state, (metrics, output if scope else None)

    return step


def init_state(rng, cfg: Conf, opt):
    params = init_fn(rng, cfg)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)
    return State(params=params, opt_state=opt_state, emas=emas)


def train(rng, cfg: Conf, ds, scope=False, log=False):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=0.9, b2=0.98)  # @nanda2023
    state = init_state(rng, cfg, opt)
    step = step_fn(ds, cfg, opt, scope)
    rngs = random.split(rng, cfg.epochs)
    state, (metrics, outputs) = lax.scan(step, state, (jnp.arange(cfg.epochs), rngs))
    log_fn(cfg, ds, metrics) if log else None
    return state, metrics, outputs


# Evaluation #########################################################################
def evaluate_fn(ds, cfg: Conf, apply, loss_fn):
    f1_score = vmap(f1_score_fn, in_axes=(1, 1)) if cfg.project == "miiii" else f1_score_fn  # type: ignore
    accuracy = vmap(accuracy_fn, in_axes=(1, 1)) if cfg.project == "miiii" else accuracy_fn  # type: ignore
    pred_fn = (
        (lambda l: (nn.sigmoid(l) > 0.5).astype(int)) if cfg.project == "miiii" else lambda l: jnp.argmax(l, axis=-1)
    )

    def aux_fn(logits, y, loss):
        pred = pred_fn(logits)
        f1, acc = f1_score(pred, y), accuracy(pred, y)
        return Split(loss=loss, f1=f1, acc=acc)

    def evaluate(params, key, train_loss, train_logits):
        valid_output = apply(params, key, ds.valid.x, cfg.dropout)
        valid_loss = loss_fn(valid_output.logits, ds.valid.y, ds.info.alpha)

        valid_metrics = aux_fn(valid_output.logits, ds.valid.y, valid_loss)
        train_metrics = aux_fn(train_logits, ds.train.y, train_loss)

        metrics = Metrics(train=train_metrics, valid=valid_metrics), valid_output  # also return the validation activations
        return metrics  # tree.map(lambda x: x.astype(jnp.float16), metries)  # store as float16

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

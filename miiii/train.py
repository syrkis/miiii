# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from miiii.model import Params, apply_fn, init_fn, Activation
from miiii.tasks import Dataset
from miiii.utils import Conf
from jax import random, value_and_grad, vmap, nn, lax, tree, jit
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import optax
from functools import partial
from typing import Tuple
from chex import dataclass, Array


# %% Types
@dataclass
class State:
    params: Params
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


# Train
@partial(vmap, in_axes=(1, 1, 0, None))  # type: ignore vmap across task (not sample)
def focal_loss_fn(logits, y, alpha, gamma):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.sigmoid_focal_loss(logits, y, alpha, gamma).astype(jnp.float32).mean()  # mean across samples


def cross_entropy_loss_fn(logits, y):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).astype(jnp.float32).mean()


def update_fn(opt, ds: Dataset, cfg: Conf):
    apply = apply_fn(cfg)
    loss_fn = focal_loss_fn if cfg.project == "miiii" else cross_entropy_loss_fn

    @jit
    def update(state, key):
        loss, losses, output, grads = grad_fn(state.params, key, ds, cfg, apply, loss_fn)
        grads, emas = filter_fn(grads, state.emas, cfg.alpha, cfg.lamb)  # grokfast values
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = State(params=params, emas=emas, opt_state=opt_state)  # type: ignore
        return state, (loss, losses, output)

    return update, apply, loss_fn


def grad_fn(params: Params, rng, ds: Dataset, cfg: Conf, apply, loss_fn) -> Tuple[Array, Array, Activation, Array]:

    @jit
    def loss_and_logits(params: Params) -> Tuple[jnp.ndarray, Tuple[Array, Activation]]:
        acts: Activation = apply(params, rng, ds.train[0], cfg.dropout)
        losses = loss_fn(acts.logits, ds.train[1], 1 - ds.train[1].mean(axis=0), cfg.gamma)
        return losses.mean(), (losses, acts)

    (loss, (losses, acts)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return loss, losses, acts, grads


@jit
def filter_fn(grads, emas, alpha: float, lamb: float):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds: Dataset, cfg: Conf, opt, scope):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2)
    update, apply, loss_fn = update_fn(opt, ds, cfg)
    evaluate = evaluate_fn(ds, cfg, apply, loss_fn)

    consort = lambda x, y: jnp.concat((x, y))[jnp.argsort(ds.idxs)].astype(jnp.float16)
    output_fn = (lambda x, y: tree.map(consort, x, y)) if scope else lambda *_: None

    @jit
    @scan_tqdm(cfg.epochs)
    def step(state, args):
        epoch, key = args
        state, (loss, losses, train_out) = update(state, key)
        metrics, valid_out = evaluate(state.params, key, losses, train_out.logits)
        return state, (metrics, output_fn(train_out, valid_out))

    return step


def init_state(rng, cfg: Conf, ds, opt):
    params: Params = init_fn(rng, cfg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)
    return State(params=params, opt_state=opt_state, emas=emas)


def train(rng, cfg: Conf, ds: Dataset, scope=False) -> Tuple[State, Tuple[Metrics, Activation | None]]:
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=0.9, b2=0.98)  # @nanda2023
    state = init_state(rng, cfg, ds, opt)
    step = step_fn(ds, cfg, opt, scope)
    rngs = random.split(rng, cfg.epochs)
    xs = (jnp.arange(cfg.epochs), rngs)
    state, output = lax.scan(step, state, xs)
    return state, output


# Evaluate
def evaluate_fn(ds: Dataset, cfg: Conf, apply, loss_fn):
    f1_score = vmap(f1_score_fn, in_axes=(1, 1)) if cfg.project == "miiii" else f1_score_fn  # type: ignore
    accuracy = vmap(accuracy_fn, in_axes=(1, 1)) if cfg.project == "miiii" else accuracy_fn  # type: ignore
    pred_fn = (
        (lambda x: (nn.sigmoid(x) > 0.5).astype(jnp.int8))
        if cfg.project == "miiii"
        else lambda x: jnp.argmax(x, axis=-1)
    )

    def aux_fn(logits, y, loss):
        pred = pred_fn(logits)
        f1, acc = f1_score(pred, y), accuracy(pred, y)
        return Split(loss=loss, f1=f1, acc=acc)

    def evaluate(params, key, train_loss, train_logits):
        valid_output = apply(params, key, ds.valid[0], cfg.dropout)
        valid_loss = loss_fn(valid_output.logits, ds.valid[1], 1 - ds.train[1].mean(axis=0), cfg.gamma)

        valid_metrics = aux_fn(valid_output.logits, ds.valid[1], valid_loss)
        train_metrics = aux_fn(train_logits, ds.train[1], train_loss)

        metrics = (Metrics(train=train_metrics, valid=valid_metrics), valid_output)
        return tree.map(lambda x: x.astype(jnp.float16), metrics)  # store as float16

    return evaluate


def accuracy_fn(y_pred, y_true):
    y_pred, y_true = y_pred.flatten(), y_true.flatten()
    return (y_pred == y_true).mean()


def f1_score_fn(y_pred, y_true):
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1
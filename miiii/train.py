# %% train.py
#   miiii train
# by: Noah Syrkis

# Imports
from functools import partial
from typing import cast

import jax.numpy as jnp
import optax
from einops import rearrange
from jax import jit, lax, random, tree, value_and_grad, vmap, debug
from jax.numpy import fft
from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii.model import apply_fn, init_fn
from miiii.tasks import Dataset
from miiii.types import Conf, Metrics, Params, MetricSplit, State

# Constants
ADAM_BETA1 = 0.9  # @nanda2023
ADAM_BETA2 = 0.98  # @nanda2023


# %% Functions
def train_fn(rng, cfg: Conf, arg, ds: Dataset):
    state, opt = init_state(rng, cfg, arg, ds)
    interval_fn = make_interval_fn(cfg, arg, opt, ds)

    inputs = (jnp.arange(arg.tick), random.split(rng, arg.tick))
    state, (scope, metrics, loss) = lax.scan(interval_fn, state, inputs)
    return state, (scope, metrics, loss)


def make_update_fn(opt, grad_fn, ds: Dataset, cfg, arg):
    def update_fn(state, key):
        loss, grads = grad_fn(state.params, key)
        grads, emas = filter_fn(grads, state.emas, cfg.lamb, cfg.alpha)
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = cast(Params, optax.apply_updates(state.params, updates))
        state = State(params=params, emas=emas, opt_state=opt_state)
        return state, loss

    return update_fn


def make_grad_fn(ds: Dataset, cfg: Conf, arg, apply, loss_fn):
    mask, weight = ds.mask, ds.weight

    @value_and_grad
    def grad_fn(params: Params, rng) -> Array:
        logits, _ = apply(rng, params, ds.train[0])
        losses = loss_fn(logits, ds.train[1], mask) * weight
        return losses.mean()

    return grad_fn


@jit
def filter_fn(grads: Params, emas: Params, lamb: float, alpha: float):
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def make_interval_fn(cfg, arg, opt, ds: Dataset):
    train_apply = apply_fn(cfg, ds, dropout=cfg.dropout)
    valid_apply = partial(apply_fn(cfg, ds, dropout=0.0), random.PRNGKey(0))
    loss_fn = vmap(cross_entropy, in_axes=(1, 1, 0))
    grad_fn = make_grad_fn(ds, cfg, arg, train_apply, loss_fn)
    update_fn = make_update_fn(opt, grad_fn, ds, cfg, arg)
    scope_fn = make_scope_fn(valid_apply, cfg, ds)

    eval_fn = make_eval_fn(ds, cfg, arg, loss_fn)

    @scan_tqdm(arg.tick)
    def interval_fn(state, inputs):
        epoch, rng = inputs
        keys = random.split(rng, cfg.epochs // arg.tick)
        state, loss = lax.scan(update_fn, state, keys)
        return state, (scope_fn(state.params), eval_fn(state), loss)

    return interval_fn


def make_scope_fn(apply_fn, cfg, ds: Dataset):
    @jit
    def scope_fn(params: Params):
        _, neurs = apply_fn(params, ds.x)
        return neurs
        neurs = rearrange(neurs[:, 0, 0, :], "(a b) n -> b a n", a=cfg.p)
        freqs = jnp.abs(fft.fft2(neurs)[1:, 1:])
        freqs = freqs / freqs.max(axis=(0, 1), keepdims=True)
        return freqs
        # mu = freqs.mean(axis=(0, 1), keepdims=True)
        # sigma = freqs.std(axis=(0, 1), keepdims=True)
        # thresh = mu + 2 * sigma
        # z = jnp.where(freqs > thresh, 1, 0)
        # print(z.shape)
        # print(neurs.shape, freqs.shape)
        # exit()
        # return z

    return scope_fn


def init_state(rng, cfg: Conf, arg, ds: Dataset):
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)  # @nanda2023
    params = init_fn(rng, cfg, arg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = cast(Params, opt.init(params))  # type: ignore
    return State(params=params, opt_state=opt_state, emas=emas), opt


def cross_entropy(logits, y, mask):
    logits = logits.astype(jnp.float64)  # enable with some jax bullshit to avoid slingshot
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()


def make_eval_fn(ds: Dataset, cfg: Conf, arg, loss_fn):
    apply = partial(apply_fn(cfg, ds, dropout=cfg.dropout), random.PRNGKey(0))
    metrics_fn = make_metrics_fn(apply, loss_fn, arg, ds)

    @jit
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

    @jit
    def metrics_fn(params, x, y):
        logits, _ = apply_fn(params, x)
        losses = loss_fn(logits, y, ds.mask) * ds.weight
        accuracy = acc_fn(logits.argmax(-1), y)
        return MetricSplit(loss=losses, acc=accuracy)

    return metrics_fn

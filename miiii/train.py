# %% train.py
#   miiii train
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple, Callable

import jax.numpy as jnp
import optax
from einops import rearrange
from optax import softmax_cross_entropy_with_integer_labels as loss_fn
from jax import lax, random, tree, value_and_grad, vmap, jit
from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii import model
from miiii.tasks import Dataset
from miiii.types import Params, Scope, State


# %% Functions
def filter_fn(grad: Params, emas: Params, lamb, alpha) -> Tuple[Params, Params]:
    emas: Params = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grad, emas)
    grad: Params = tree.map(lambda grad, ema: grad + lamb * ema, grad, emas)
    return grad, emas


def train_fn(rng, cfg, ds: Dataset, state: State, opt, mask: Array) -> Tuple[State, Tuple[Array, Scope]]:
    @scan_tqdm(cfg.tick)
    def aux(state: State, inputs: Tuple[Array, Array]) -> Tuple[State, Tuple[Array, Scope]]:
        keys = random.split(inputs[1], cfg.epochs // cfg.tick)
        state, loss = lax.scan(jit(partial(update_fn, opt, ds, cfg, mask)), state, keys)
        return state, (loss, jit(partial(scope_fn, ds, cfg, rng))(state))

    return lax.scan(aux, state, (jnp.arange(cfg.tick), random.split(rng, cfg.tick)))


def scope_fn(ds: Dataset, cfg, rng: Array, state) -> Scope:
    apply: Callable[[Array], Tuple[Array, Array]] = vmap(partial(model.apply, state.params, 0.0, rng))
    train, valid = apply(ds.train.x), apply(ds.valid.x)
    train_acc = (train[0].argmax(-1) == ds.train.y).mean(0)
    valid_acc = (valid[0].argmax(-1) == ds.valid.y).mean(0)
    train_cce = loss_fn(train[0], ds.train.y, where=ds.classes).mean(0)
    valid_cce = loss_fn(valid[0], ds.valid.y, where=ds.classes).mean(0)
    neu = rearrange(jnp.concat((train[1], valid[1]))[:, -1][ds.idxs.argsort()], "(a b) n -> n a b", a=cfg.p)[::16, :23, :23]
    return Scope(train_acc=train_acc, valid_acc=valid_acc, train_cce=train_cce, valid_cce=valid_cce, neu=neu)


def update_fn(opt, ds: Dataset, cfg, mask, state: State, rng) -> Tuple[State, Array]:
    loss, grad = grad_fn(state.params, cfg, rng, ds.train.x, ds.train.y, ds.classes, ds.weight, mask)
    grad, emas = filter_fn(grad, state.emas, cfg.lamb, cfg.alpha)
    updates, opt_state = opt.update(grad, state.opt_state, state.params)
    params: Params = optax.apply_updates(state.params, updates)  # type: ignore
    return State(params=params, opt_state=opt_state, emas=emas), loss


@value_and_grad
def grad_fn(params: Params, cfg, rng, x: Array, y: Array, classes: Array, weight: Array, mask) -> Array:
    logits, z = vmap(partial(model.apply, params, cfg.dropout, rng))(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(jnp.float64(logits), y, where=classes) / weight
    return (losses.mean(0) * mask).sum() / mask.sum()  # mask away some tasks


def init_fn(rng, cfg, ds: Dataset) -> Tuple[State, optax.GradientTransformation]:
    opt: optax.GradientTransformation = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=cfg.b1, b2=cfg.b2)
    params = model.init_fn(rng, cfg, ds)
    emas: Params = tree.map(lambda x: jnp.zeros_like(x), params)
    return State(params=params, opt_state=opt.init(params), emas=emas), opt  # type: ignore

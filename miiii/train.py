# %% train.py
#   miiii train
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple

import jax.numpy as jnp
import optax
from jax import lax, random, tree, value_and_grad, vmap
from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii import model
from miiii.tasks import Dataset
from miiii.types import Params, Scope, State


# %% Functions
def filter_fn(grad: Params, emas: Params, lamb, alpha) -> Tuple[Params, Params]:  # TODO: mi alg (the third comp)
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grad, emas)
    grad = tree.map(lambda grad, ema: grad + lamb * ema, grad, emas)
    return grad, emas


def train_fn(rng, cfg, ds: Dataset, state: State, opt) -> Tuple[State, Tuple[Array, Scope]]:
    @scan_tqdm(cfg.tick)
    def aux(state: State, inputs: Tuple[Array, Array]) -> Tuple[State, Tuple[Array, Scope]]:
        keys = random.split(inputs[1], cfg.epochs // cfg.tick)
        state, loss = lax.scan(partial(update_fn, opt, ds, cfg), state, keys)
        return state, (loss, scope_fn(ds, rng, state))

    return lax.scan(aux, state, (jnp.arange(cfg.tick), random.split(rng, cfg.tick)))


def scope_fn(ds: Dataset, rng: Array, state) -> Scope:
    logits: Array = vmap(partial(model.apply, state.params))(ds.x)
    acc: Array = (logits.argmax(-1) == ds.y).mean(0)
    cce = optax.softmax_cross_entropy_with_integer_labels(logits, ds.y, where=ds.mask)
    return Scope(acc=acc, cce=cce)  # type: ignore


def update_fn(opt, ds: Dataset, cfg, state: State, key) -> Tuple[State, Array]:
    loss, grad = grad_fn(state.params, ds.x, ds.y, ds.mask)
    grad, emas = filter_fn(grad, state.emas, cfg.lamb, cfg.alpha)
    updates, opt_state = opt.update(grad, state.opt_state, state.params)  # type: ignore
    params: Params = optax.apply_updates(state.params, updates)  # type: ignore
    return State(params=params, opt_state=opt_state, emas=state.emas), loss


@value_and_grad
def grad_fn(params: Params, x: Array, y: Array, mask: Array) -> Array:
    logits: Array = vmap(partial(model.apply, params))(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask)
    return losses.mean()


def init_fn(rng, cfg, ds: Dataset) -> Tuple[State, optax.GradientTransformation]:
    opt: optax.GradientTransformation = optax.adam(cfg.lr)
    params = model.init_fn(rng, cfg, ds)
    emas: Params = tree.map(lambda x: jnp.zeros_like(x), params)
    return State(params=params, opt_state=opt.init(params), emas=emas), opt  # type: ignore

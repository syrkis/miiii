# %% train.py
#   miiii train
# by: Noah Syrkis

# Imports
from functools import partial
from typing import cast

import jax.numpy as jnp
import optax
from jax import lax, random, tree, value_and_grad, vmap, debug
from typing import Tuple
from optax import GradientTransformation

# from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii import model
from miiii.tasks import Dataset
from miiii.types import Params, State, Scope


# Constants
# ADAM_BETA1 = 0.9
# ADAM_BETA2 = 0.98


# %% Functions
def train_fn(rng, cfg, ds: Dataset, state: State, opt: GradientTransformation) -> Tuple[State, Tuple[Array, Scope]]:
    # @scan_tqdm(cfg.tick)
    # debug.print("{x}", x=ds.x.deeper)

    def aux(state, idx_rng) -> Tuple[State, Tuple[Array, Scope]]:
        keys = random.split(idx_rng[1], cfg.epochs // cfg.tick)
        state, loss = lax.scan(partial(update_fn, opt, ds, cfg), state, keys)
        return state, (loss, scope_fn(ds, rng, state))

    state, (loss, acc) = lax.scan(aux, state, (jnp.arange(cfg.tick), random.split(rng, cfg.tick)))
    return state, (loss, acc)


def scope_fn(ds, rng, state) -> Scope:
    logits, z = model.apply(rng, state.params, ds.x)
    acc: Array = (logits.argmax(-1) == ds.y).mean(0)
    cce: Array = cross_entropy(logits, ds.y, ds.mask)
    # debug.print("{i}", i=cce)
    return Scope(acc=acc, cce=cce)


def update_fn(opt: GradientTransformation, ds: Dataset, cfg, state: State, key) -> Tuple[State, Array]:
    loss, grad = value_and_grad(loss_fn)(state.params, ds.x, ds.y, ds.mask, key)
    # grad, emas = filter_fn(grad, state.emas, cfg.lamb, cfg.alpha)
    updates, opt_state = opt.update(grad, state.opt_state, state.params)  # type: ignore
    params = optax.apply_updates(state.params, updates)  # type: ignore
    return State(params=params, emas=state.emas, opt_state=opt_state), loss  # type: ignore


def loss_fn(params: Params, x, y, mask, rng: Array) -> Array:
    logits, z = model.apply(rng, params, x)
    losses: Array = cross_entropy(logits, y, mask)  # / ds.task  # normalize by n-ary task
    return losses.mean()  # mean across tasks (weigted by expected n-ary classification loss)


# def filter_fn(grad: Params, emas: Params, lamb, alpha) -> Tuple[Params, Params]:  # TODO: mi alg (the third comp)
#     emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grad, emas)
#     grad = tree.map(lambda grad, ema: grad + lamb * ema, grad, emas)
#     return grad, emas


# %% INIT
def init_fn(rng, cfg, ds: Dataset) -> Tuple[State, GradientTransformation]:
    opt = optax.adam(cfg.lr)
    params = model.init_fn(rng, cfg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    return State(params=params, opt_state=opt.init(params), emas=emas), opt  # type: ignore


# %% HELPERS
@partial(vmap, in_axes=(1, 1, 0))
def cross_entropy(logits: Array, y: Array, mask: Array) -> Array:  # mean cross entropy acroess samples
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()

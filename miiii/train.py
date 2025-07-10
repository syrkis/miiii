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
from jax_tqdm import scan_tqdm
from jaxtyping import Array

from miiii import model
from miiii.tasks import Dataset
from miiii.types import Params, State


# Constants
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98


# %% Functions
def train_fn(rng, cfg, ds: Dataset, state: State, opt: GradientTransformation) -> Tuple[State, Array]:
    update = partial(update_fn, opt, ds, cfg)
    state, loss = update(state, rng)

    @scan_tqdm(cfg.tick)
    def step(state, idx_rng) -> Tuple[State, Array]:
        keys = random.split(idx_rng[1], cfg.epochs // cfg.tick)
        state, loss = lax.scan(update, state, keys)
        return state, loss

    input = (jnp.arange(cfg.tick), random.split(rng, cfg.tick))
    state, loss = lax.scan(step, state, input)

    return state, loss


def update_fn(opt: GradientTransformation, ds: Dataset, cfg, state: State, key) -> Tuple[State, Array]:
    loss, grad = value_and_grad(partial(loss_fn, ds))(state.params, key)
    grad, emas = filter_fn(grad, state.emas, cfg.lamb, cfg.alpha)
    updates, opt_state = opt.update(grad, state.opt_state, state.params)  # type: ignore
    params = optax.apply_updates(state.params, updates)  # type: ignore
    return State(params=params, emas=emas, opt_state=opt_state), loss  # type: ignore


def loss_fn(ds: Dataset, params: Params, rng) -> Array:
    logits, z = model.apply(ds, rng, params, ds.x)
    losses = cross_entropy(logits, ds.y, ds.weight)  # * ds.mask
    return losses.mean()


def filter_fn(grad: Params, emas: Params, lamb: float, alpha: float) -> Tuple[Params, Params]:
    emas = tree.map(lambda grad, ema: ema * alpha + grad * (1 - alpha), grad, emas)
    grad = tree.map(lambda grad, ema: grad + lamb * ema, grad, emas)
    return grad, emas


# %% INIT
def init_fn(rng, cfg, ds: Dataset) -> Tuple[State, GradientTransformation]:
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)
    params = model.init_fn(rng, cfg, ds)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = cast(Params, opt.init(params))  # type: ignore
    return State(params=params, opt_state=opt_state, emas=emas), opt


# %% HELPERS
@partial(vmap, in_axes=(1, 1, 0))
def cross_entropy(logits: Array, y: Array, mask: Array):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()

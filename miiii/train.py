# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from typing import Tuple

import jax.numpy as jnp
import optax
from chex import Array
from jax import jit, lax, random, tree, value_and_grad, vmap
from jax.numpy import fft
from jax_tqdm import scan_tqdm
from functools import partial
from einops import rearrange

from miiii.model import apply_fn, init_fn
from miiii.tasks import Dataset, Task
from miiii.utils import Activation, Conf, Metrics, Params, Split, State, Scope


ADAM_BETA1 = 0.9  # @nanda2023
ADAM_BETA2 = 0.98  # @nanda2023
ALPHA = 0.98


# %% Functions
def update_fn(opt, ds: Dataset, task: Task, cfg: Conf):
    train_apply = apply_fn(cfg, ds, task, eval=False)
    valid_apply = partial(apply_fn(cfg, ds, task, eval=True), random.PRNGKey(0))
    grad = grad_fn(ds, task, cfg, train_apply, task.loss_fn, task.mask)

    def update(state, key):
        loss, losses, output, grads = grad(state.params, key)
        grads, emas = filter_fn(grads, state.emas, cfg.lamb)  # grokfast values
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = State(params=params, emas=emas, opt_state=opt_state)  # type: ignore
        return state, (loss, losses, output), grads

    return update, train_apply, valid_apply


def grad_fn(ds: Dataset, task: Task, cfg: Conf, apply, loss_fn, mask):
    @jit
    def grad(params: Params, rng) -> Tuple[Array, Array, Activation, Array]:
        def loss_and_logits(params: Params) -> Tuple[jnp.ndarray, Tuple[Array, Activation]]:
            acts: Activation = apply(rng, params, ds.x.train)
            losses = loss_fn(acts.logits, ds.y.train, 1 - ds.y.train.mean(0), 2, mask) * task.weight
            return losses.mean(), (losses, acts)

        (loss, (losses, acts)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
        return loss, losses, acts, grads

    return grad


@jit
def filter_fn(grads, emas, lamb: float):
    emas = tree.map(lambda grad, ema: ema * ALPHA + grad * (1 - ALPHA), grads, emas)
    grads = tree.map(lambda grad, ema: grad + lamb * ema, grads, emas)
    return grads, emas


def step_fn(ds: Dataset, task: Task, cfg: Conf, opt, scope):
    # opt = optax.adamw(cfg.lr, weight_decay=cfg.l2)
    update, train_apply, valid_apply = update_fn(opt, ds, task, cfg)
    evaluate = evaluate_fn(ds, task, cfg, valid_apply)

    @jit
    @scan_tqdm(cfg.epochs)
    def step(state, args):
        epoch, key = args
        state, (loss, losses, train_acts), grads = update(state, key)
        metrics, scope = evaluate(state.params, grads, losses, train_acts)
        return state, (metrics, scope)  # output_fn(train_out, valid_out))

    return step


def init_state(rng, cfg: Conf, ds: Dataset, task: Task, opt):
    params: Params = init_fn(rng, cfg, ds, task)
    emas = tree.map(lambda x: jnp.zeros_like(x), params)
    opt_state = opt.init(params)
    return State(params=params, opt_state=opt_state, emas=emas)


def train(rng, cfg: Conf, ds: Dataset, task: Task, scope=False) -> Tuple[State, Tuple[Metrics, Activation | None]]:
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2, b1=ADAM_BETA1, b2=ADAM_BETA2)  # @nanda2023
    state = init_state(rng, cfg, ds, task, opt)
    step = step_fn(ds, task, cfg, opt, scope)
    rngs = random.split(rng, cfg.epochs)
    xs = (jnp.arange(cfg.epochs), rngs)
    state, output = lax.scan(step, state, xs)
    return state, output


# Evaluate
def evaluate_fn(ds: Dataset, task: Task, cfg: Conf, apply):
    scope = scope_fn(ds, cfg, apply)

    @partial(vmap, in_axes=((1, 1) if task.span == "factors" else (None, None)))
    def acc_fn(y_pred, y_true):
        return (y_pred == y_true).mean()

    @jit
    def evaluate(params, grads, train_loss, train_acts):
        valid_acts = apply(params, ds.x.eval)
        valid_loss = task.loss_fn(valid_acts.logits, ds.y.eval, 1 - ds.y.train.mean(0), 2, task.mask) * task.weight

        valid_metrics = Split(loss=valid_loss, acc=acc_fn(valid_acts.logits.argmax(-1), ds.y.eval))
        train_metrics = Split(loss=train_loss, acc=acc_fn(train_acts.logits.argmax(-1), ds.y.train))

        metrics = Metrics(train=train_metrics, valid=valid_metrics)
        return tree.map(lambda x: x.astype(jnp.float16), metrics), scope(params, grads, train_acts, valid_acts)

    return evaluate


def scope_fn(ds, cfg, apply):
    fn = lambda a, b: rearrange(jnp.concat((a, b))[ds.udxs].squeeze(), "(a b) ... -> a b ...", a=cfg.p, b=cfg.p)  # noqa

    def scope_aux(params, grads, train_acts, valid_acts):
        acts = tree.map(fn, train_acts, valid_acts)
        grad_norms = tree.map(lambda x: jnp.linalg.norm(x), grads)
        neurs = jnp.abs(fft.rfft2(rearrange(acts.ffwd.squeeze()[:, -1], "(x0 x1) n -> n x0 x1")))[..., 1:, 1:]
        freqs = ((neurs / neurs.max()) > 0.5).sum((0, 1))
        scope = Scope(grad_norms=grad_norms, neuron_freqs=freqs)
        return tree.map(lambda x: x.astype(jnp.float16), scope)

    return scope_aux

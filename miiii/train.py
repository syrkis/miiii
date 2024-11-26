# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
from typing import Tuple

import jax.numpy as jnp
import optax
from chex import Array
from jax import jit, lax, nn, random, tree, value_and_grad, vmap
from jax_tqdm import scan_tqdm
from functools import partial

from miiii.model import apply_fn, init_fn
from miiii.tasks import Dataset, Task
from miiii.utils import Activation, Conf, Metrics, Params, Split, State


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
            losses = loss_fn(acts.logits, ds.y.train, 1 - ds.y.train.mean(0), 2, mask) / task.weight
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
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2)
    update, train_apply, valid_apply = update_fn(opt, ds, task, cfg)
    evaluate = evaluate_fn(ds, task, cfg, valid_apply)

    consort = lambda x, y: jnp.concat((x, y))[jnp.argsort(ds.idxs)].astype(jnp.float16)  # noqa
    output_fn = (lambda x, y: tree.map(consort, x, y)) if scope else lambda *_: None

    @jit
    @scan_tqdm(cfg.epochs)
    def step(state, args):
        epoch, key = args
        state, (loss, losses, train_out), grads = update(state, key)
        metrics, valid_out = evaluate(state.params, grads, losses, train_out.logits)
        return state, (metrics, output_fn(train_out, valid_out))

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
    # f1_score = vmap(f1_score_fn, in_axes=(1, 1)) if task.span == "factors" else f1_score_fn
    accuracy = vmap(accuracy_fn, in_axes=(1, 1)) if task.span == "factors" else accuracy_fn

    # TODO: report norm of all params through training

    pred_fn = (lambda x: x.argmax(-1)) if task.type == "remainder" else lambda x: (nn.sigmoid(x) > 0.5).astype(jnp.int8)

    def aux_fn(logits, y, loss):
        pred = pred_fn(logits)
        # f1, acc = f1_score(pred, y), accuracy(pred, y)
        acc = accuracy(pred, y)
        return Split(loss=loss, acc=acc)

    @jit
    def evaluate(params, grads, train_loss, train_logits):
        valid_output = apply(params, ds.x.eval)
        valid_loss = task.loss_fn(valid_output.logits, ds.y.eval, 1 - ds.y.train.mean(0), 2, task.mask) / task.weight

        valid_metrics = aux_fn(valid_output.logits, ds.y.eval, valid_loss)
        train_metrics = aux_fn(train_logits, ds.y.train, train_loss)

        metrics = Metrics(train=train_metrics, valid=valid_metrics, grads=tree.map(lambda x: jnp.linalg.norm(x), grads))
        # return metrics, valid_output
        return tree.map(lambda x: x.astype(jnp.float16), metrics), valid_output

    return evaluate


# @jit
def accuracy_fn(y_pred, y_true):
    return (y_pred == y_true).mean()


@jit
def f1_score_fn(y_pred, y_true):
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

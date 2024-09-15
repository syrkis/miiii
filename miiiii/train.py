# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax
from jax import random, jit, value_and_grad, vmap
import optax
import jax.numpy as jnp
from functools import partial
from aim import Run, Figure


# functions
@partial(vmap, in_axes=(1, 1, 0))  # TODO: <- double check that axis are correct
def focal_loss(logits, y, alpha):
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha)


def make_update_fn(opt):
    @jit
    def update_fn(params, grads, opt_state):
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_fn


def make_grad_fn(loss_fn, apply_fn, cfg):
    @jit
    def grad_fn(state, rng, x, y, alpha):  # maybe add allow_int flag below
        def loss_and_logits(params):
            logits = apply_fn(params, rng, x, cfg.dropout)
            loss = loss_fn(logits, y, alpha).mean()
            return loss, logits

        (loss, logits), grads = value_and_grad(loss_and_logits, has_aux=True)(state.params)
        grads, state = gradfilter_ema(grads, state)  # @lee2024b (grokfast)
        return loss, grads, logits, state

    return grad_fn


@partial(jit, static_argnums=(2,))
def gradfilter_ema(grads, state: mi.kinds.State, alpha=0.98, lamb=2.0):
    # @lee2024b grokfast-like EMA gradient filtering
    def _update_ema(prev_ema, gradient):
        return prev_ema * alpha + gradient * (1 - alpha)

    def _apply_ema(gradient, ema):
        return gradient + ema * lamb

    ema_grads = jax.tree.map(_update_ema, state.ema_grads, grads)
    filtered_grads = jax.tree.map(_apply_ema, grads, ema_grads)
    state = mi.kinds.State(ema_grads=ema_grads, opt_state=state.opt_state, params=state.params)
    return filtered_grads, state


def make_step_fn(grad_fn, update_fn, ds: mi.kinds.Dataset, eval_fn):
    @jit
    def step_fn(state, rng):
        params, opt_state = state.params, state.opt_state
        rng, key = random.split(rng)
        loss, grads, logits, state = grad_fn(state, key, ds.train.x, ds.train.y, ds.info.alpha)
        params, opt_state = update_fn(params, grads, opt_state)
        metrics = eval_fn(params, rng, loss, logits)
        state = mi.kinds.State(params=params, opt_state=opt_state, ema_grads=state.ema_grads)
        return state, metrics

    return step_fn


def make_eval_fn(apply_fn, loss_fn, ds):
    @jit
    def eval_fn(params, rng, train_loss, train_logits):
        valid_logits = apply_fn(params, rng, ds.valid.x, 0.0)
        valid_loss = loss_fn(valid_logits, ds.valid.y, ds.info.alpha).mean()  # number

        train_metrics = metrics_fn(ds.train.y, train_logits)
        valid_metrics = metrics_fn(ds.valid.y, valid_logits)

        return dict(
            train_loss=train_loss,
            valid_loss=valid_loss,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
        )

    return eval_fn


def predict_fn(logits):
    return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)


def metrics_fn(y_true, logits):
    """returns accuracy, precision, recall, f1"""
    y_pred = predict_fn(logits)
    tp = jnp.sum(y_true * y_pred, axis=0)
    fp = jnp.sum((1 - y_true) * y_pred, axis=0)
    fn = jnp.sum(y_true * (1 - y_pred), axis=0)
    acc = (y_true == y_pred).mean(axis=0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    loss = optax.sigmoid_focal_loss(logits, y_true).mean(axis=0)
    return dict(loss=loss, f1=f1)  # , prec=prec, rec=rec, acc=acc)


def make_train_fn(step_fn):
    def train_fn(steps, rng, state):
        rngs = random.split(rng, steps)
        state, metrics = jax.lax.scan(step_fn, state, rngs, length=steps)
        return state, flatten_metrics(metrics)

    def flatten_metrics(metrics):
        """make metrics into flate table"""
        valid_metrics = {k: v.T for k, v in metrics["valid_metrics"].items()}
        train_metrics = {k: v.T for k, v in metrics["train_metrics"].items()}
        metrics = {"train": train_metrics, "valid": valid_metrics}
        return metrics

    return train_fn


def init_train(apply_fn, params, cfg, ds: mi.kinds.Dataset):
    loss_fn = focal_loss  # @nanda2023
    opt = optax.adamw(cfg.lr, weight_decay=cfg.l2)  # @nanda2023
    opt_state = opt.init(params)

    update_fn = make_update_fn(opt)
    grad_fn = make_grad_fn(loss_fn, apply_fn, cfg)

    ema_grads = jax.tree.map(jnp.zeros_like, params)
    state = mi.kinds.State(params=params, opt_state=opt_state, ema_grads=ema_grads)

    eval_fn = make_eval_fn(apply_fn, loss_fn, ds)
    step_fn = make_step_fn(grad_fn, update_fn, ds, eval_fn)

    train_fn = make_train_fn(step_fn)
    return train_fn, state

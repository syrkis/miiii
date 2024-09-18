# %% train.py
#   miiii train
# by: Noah Syrkis

# %% Imports
import miiiii as mi
import jax
from jax import random, jit, value_and_grad, vmap, nn, lax, tree
import optax
import jax.numpy as jnp
from functools import partial
from aim import Run, Figure
from jax_tqdm import scan_tqdm
from tqdm import tqdm


# functions
@partial(vmap, in_axes=(1, 1, 0))  # vmap across task (not sample)
def loss_fn(logits, y, alpha):
    return optax.sigmoid_focal_loss(logits, y, alpha=alpha)  # do not take mean (task loss vector)


def update_fn(opt, ds, cfg):
    def update(params, opt_state, key):
        losses, logits, grads = mi.train.grad_fn(params, key, ds, cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, losses, logits

    return update


def grad_fn(params, rng, ds, cfg):  # maybe add allow_int flag below
    def loss_and_logits(params):
        logits = mi.model.apply(params, rng, ds.train.x, cfg.dropout)
        losses = loss_fn(logits, ds.train.y, ds.info.alpha)  # mean for optimization
        return losses.mean(), (losses, logits)

    (_, (losses, logits)), grads = value_and_grad(loss_and_logits, has_aux=True)(params)
    return losses, logits, grads


def step_fn(ds, cfg):
    opt = optax.adam(cfg.lr)
    evaluate = evaluate_fn(ds, cfg)
    update = update_fn(opt, ds, cfg)

    @jit
    def step(state, key):
        params, opt_state = state
        params, opt_state, losses, logits = update(params, opt_state, key)
        metrics = evaluate(params, key, logits, losses)
        mi.utils.check_nan(params, "params")
        mi.utils.check_nan(metrics, "metrics")
        return (params, opt_state), metrics

    return step


def train(rng, cfg, ds):
    params = mi.model.init_fn(rng, cfg)
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)  # type: ignore
    step = step_fn(ds, cfg)
    rngs = random.split(rng, cfg.epochs)
    state = (params, opt_state)
    state, metrics = lax.scan(step, state, rngs)  # getting a weird error
    # metrics = []
    # for key in (pbar := tqdm(rngs)):
    #     state, metric = step(state, key)
    #     metrics.append(metric)
    #     pbar.set_description(
    #         f"train_loss: {metric['train']['losses'].mean():.3f}, valid_loss: {metric['valid']['losses'].mean():.3f}"
    #     )
    # matrics = tree.map(lambda x, y: jnp.append(x, y), metrics, metric)
    # mi.utils.check_nan(metrics, "metrics")
    # mi.utils.check_nan(state[0], "params")
    # mi.utils.check_nan(state[1], "opt_state")

    return state, metrics


# Evaluation #########################################################################
def evaluate_fn(ds, cfg):
    def evaluate(params, key, train_logits, train_losses):
        valid_logits = mi.model.apply(params, key, ds.valid.x, cfg.dropout)
        valid_losses = mi.train.loss_fn(valid_logits, ds.valid.y, ds.info.alpha)
        train_f1 = f1_score(train_logits, ds.train.y)
        valid_f1 = f1_score(valid_logits, ds.valid.y)
        train_metrics = dict(losses=train_losses, f1=train_f1)
        valid_metrics = dict(losses=valid_losses, f1=valid_f1)
        return dict(train=train_metrics, valid=valid_metrics)

    return evaluate


@partial(vmap, in_axes=(1, 1))
def f1_score(y_logits, y_true):
    y_pred = (nn.sigmoid(y_logits) > 0.5).astype(jnp.int32)  # 0.5 threshold
    confusion = jnp.eye(2)[y_pred].T @ jnp.eye(2)[y_true]
    tp, fp, fn = confusion[1, 1], confusion[1, 0], confusion[0, 1]
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

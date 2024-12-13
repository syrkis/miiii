# %% scope.py
#    miiii scope functions
# by: Noah Syrkis

# Imports
from typing import Tuple
from miiii.utils import Conf, Metrics, Params, Split, State
from miiii.tasks import Dataset
from jax import lax, random, tree, value_and_grad, vmap, jit
from functools import partial
from miiii.model import apply_fn, init_fn


def make_scope_fn(ds, cfg, arg, loss_fn):
    eval_fn = make_eval_fn(ds, cfg, arg, loss_fn)

    def scope_fn(state):
        metrics = eval_fn(state)
    return eval_fn

# Functions
def make_eval_fn(ds: Dataset, cfg: Conf, arg, loss_fn):
    apply = partial(apply_fn(cfg, ds, dropout=cfg.dropout), random.PRNGKey(0))
    metrics_fn = make_metrics_fn(apply, loss_fn, arg, ds)

    @jit
    def eval_fn(state: State):
        valid_metrics = metrics_fn(state.params, ds.x.eval, ds.y.eval)
        train_metrics = metrics_fn(state.params, ds.x.train, ds.y.train)
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
        logits = apply_fn(params, x)
        losses = loss_fn(logits, y, ds.mask) * ds.weight
        accuracy = acc_fn(logits.argmax(-1), y)
        return Split(loss=losses, acc=accuracy)
    return metrics_fn

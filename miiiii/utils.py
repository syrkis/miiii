# %% utils.py
#   miiii utils
# by: Noah Syrkis

# %% imports
import os
import jax.numpy as jnp
from jax import tree
import yaml
import pickle
from aim import Run
from chex import dataclass


# %% constants
red = "#da3527"
blue = "#002fa7"


def check_nan(pytree, name):
    return jnp.array(tree.flatten(tree.map(lambda x: jnp.isnan(x).any(), pytree))[0]).any()


@dataclass
class Conf:
    vocab_size: int  # compute from p
    seq_len: int  # compute from p
    n: int  # compute from p
    base: int

    latent_dim: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 1000
    lr: float = 1e-3
    l2: float = 0.1
    dropout: float = 0.1
    p: int = 113
    # task: str = "prime"  # "prose"
    # block: str = "vaswani"
    # causal: bool = False


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
def cfg_fn(kwargs):
    n = kwargs["p"] ** 2
    seq_len = digit_fn(n, kwargs["p"]).item()
    base = kwargs["p"]
    cfg = Conf(**kwargs | {"seq_len": seq_len, "n": n, "vocab_size": kwargs["p"], "base": base})
    return cfg


def save_params(params, fname):
    path = os.path.join("data", fname)
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_params(fname):
    path = os.path.join("data", fname)
    with open(path, "rb") as file:
        return pickle.load(file)


def track_metrics(metrics, ds, cfg):
    run = Run(experiment="miiiii")
    run["cfg"] = cfg.__dict__

    for epoch in range(cfg.epochs):
        for idx, task in enumerate(ds.info.tasks):
            for split in ["train", "valid"]:
                to_log = {k: v[epoch][idx] for k, v in metrics[split].items()}
                run.track(to_log, epoch=epoch + 1, context={"task": task, "split": split})


def metrics_to_dict(metrics):
    train_metrics = dict(loss=metrics.train_loss.T, f1=metrics.train_f1.T)
    valid_metrics = dict(loss=metrics.valid_loss.T, f1=metrics.valid_f1.T)
    return dict(train=train_metrics, valid=valid_metrics)

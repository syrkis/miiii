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
    # task is either "prime" or "prose"
    vocab_size: int
    batch_size: int
    seq_len: int
    task: str = "prime"  # "prose"
    causal: bool = False
    base: int = 2
    n: int = 1024
    latent_dim: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 100
    lr: float = 1e-3
    block: str = "vaswani"
    l2: float = 0.1
    dropout: float = 0.1


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
def cfg_fn(
    base=37,
    n=1024,  # 113 ^ 2 @nanda2023 shoutout
    epochs=10000,
    lr=3e-4,
    dropout=0.5,
    latent_dim=64,
    heads=4,
    depth=4,
    task="prime",
    batch_size=32,
    seq_len=32,
    l2=1.0,
):
    vocab_size = base if task == "prime" else 118  # hardocded vocab size of borges' ficciones
    seq_len = digit_fn(n, base).item() if task == "prime" else seq_len
    cfg = Conf(
        batch_size=batch_size,  # only used for prose
        causal=True if task == "prose" else False,
        base=base,
        n=n,  # Number of
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        latent_dim=latent_dim,
        heads=heads,
        depth=depth,
        task=task,
        vocab_size=vocab_size,
        seq_len=seq_len,
        l2=l2,
    )
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

# %% utils.py
#   miiii utils
# by: Noah Syrkis

# %% imports
import miiiii.kinds as kinds
import miiiii.prime as prime
import argparse
import os
import jax.numpy as jnp
from jax import tree
import yaml
import pickle
from aim import Run


# %% constants
red = "#da3527"
blue = "#002fa7"


def check_nan(pytree, name):
    return jnp.array(tree.flatten(tree.map(lambda x: jnp.isnan(x).any(), pytree))[0]).any()


# %% functions
def cfg_fn(
    base=37,
    n=1024,  # 113 ^ 2 @nanda2023 shoutout
    epochs=1000,
    lr=3e-4,
    dropout=0.1,
    latent_dim=64,
    heads=8,
    depth=4,
    task="prime",
    batch_size=32,
    seq_len=32,
    l2=0.1,
):
    vocab_size = base if task == "prime" else 118  # hardocded vocab size of borges' ficciones
    seq_len = prime.digit_fn(n, base).item() if task == "prime" else seq_len
    cfg = kinds.Conf(
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

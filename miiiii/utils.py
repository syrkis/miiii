# %% utils.py
#   miiii utils
# by: Noah Syrkis

# %% imports
import os
import jax.numpy as jnp
from jax import tree
import yaml
import pickle
import wandb
import sqlite3
from pathlib import Path
import numpy as np
from aim import Run
from tqdm import tqdm


# from aim import Run
from chex import dataclass
from typing import Literal


# %% constants
red = "#da3527"
blue = "#002fa7"


def check_nan(pytree, name):
    return jnp.array(tree.flatten(tree.map(lambda x: jnp.isnan(x).any(), pytree))[0]).any()


@dataclass
class Conf:
    project: str = "miiii"
    prime: int = 113
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 1000
    lr: float = 1e-3
    l2: float = 0.1
    dropout: float = 0.1
    train_frac: float = 0.3  # @nanda2023

    # base: int
    # power: int = 2 # if we should use a different base.
    # task: str = "prime"  # "prose"
    # block: str = "vaswani"
    # causal: bool = False


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
def cfg_fn(kwargs, hyper_kwargs={}):
    cfg = Conf(**kwargs, **hyper_kwargs)
    return cfg


def save_params(params, fname):
    path = os.path.join("data", fname)
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_params(fname):
    path = os.path.join("data", fname)
    with open(path, "rb") as file:
        return pickle.load(file)


def metrics_to_dict(metrics):
    train_metrics = dict(loss=metrics.train_loss.T, f1=metrics.train_f1.T)
    valid_metrics = dict(loss=metrics.valid_loss.T, f1=metrics.valid_f1.T)
    return dict(train=train_metrics, valid=valid_metrics)


def log_fn(cfg: Conf, ds, metrics):
    # Initialize Aim run
    run = Run(
        experiment=cfg.project,
        system_tracking_interval=None,  # Disable system tracking
    )

    # Log config
    run["hparams"] = cfg.__dict__

    metrics_arrays = {
        "loss": (np.array(metrics.train.loss), np.array(metrics.valid.loss)),
        "f1": (np.array(metrics.train.f1), np.array(metrics.valid.f1)),
        "acc": (np.array(metrics.train.acc), np.array(metrics.valid.acc)),
    }

    # Log metrics for each epoch
    for epoch in tqdm(range(cfg.epochs)):
        for metric_name, (train_values, valid_values) in metrics_arrays.items():
            # Log individual task metrics
            for i, task in enumerate(ds.info.tasks if cfg.project == "miiii" else range(1)):
                # Train metrics
                run.track(
                    train_values[epoch, i] if cfg.project == "miiii" else train_values[epoch],
                    name=metric_name,
                    epoch=epoch,
                    context={
                        "project": cfg.project,
                        "split": "train",
                        "task": task,
                    },
                )
                # Valid metrics
                run.track(
                    valid_values[epoch, i] if cfg.project == "miiii" else valid_values[epoch],
                    name=metric_name,
                    epoch=epoch,
                    context={
                        "project": cfg.project,
                        "split": "valid",
                        "task": task,
                    },
                )

    run.close()


def name_run_fn(cfg: Conf) -> str:
    """Create a descriptive run name from configuration.
    Format: task_ld{latent_dim}_de{depth}_he{heads}_lr{lr}_l2{l2}_dr{dropout}
    Example: miiii_ld128_de2_he4_lr1e-3_l20.1_dr0.1
    """
    return (
        # f"{cfg.task}"
        f"_pm{cfg.prime}"
        f"_ld{cfg.latent_dim}"
        f"_de{cfg.depth}"
        f"_he{cfg.heads}"
        f"_lr{cfg.lr:g}"  # :g removes trailing zeros
        f"_l2{cfg.l2:g}"
        f"_dr{cfg.dropout:g}"
        f"_ep{cfg.epochs}"
    )

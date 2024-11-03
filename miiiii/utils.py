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
from functools import partial


# from aim import Run
from chex import dataclass
from typing import Literal


@dataclass
class Conf:
    project: str = "nanda"
    alpha: float = 0.98  # not sure what this does (grokfast)
    lamb: float = 2  # set to 0 for no filter (grokfast)
    prime: int = 113
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 10000
    lr: float = 1e-3
    l2: float = 1.0
    dropout: float = 0.5
    train_frac: float = 0.3  # @nanda2023

def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
def cfg_fn(kwargs):
    return Conf(**kwargs)


def metrics_to_dict(metrics):
    return {
        "loss": {"train": np.array(metrics.train.loss), "valid" : np.array(metrics.valid.loss)},
        "f1": {"train": np.array(metrics.train.f1), "valid" : np.array(metrics.valid.f1)},
        "acc": {"train": np.array(metrics.train.acc), "valid" : np.array(metrics.valid.acc)},
    }

def log_split(run, cfg, metrics, epoch, task, task_idx, split):
    fn = partial(log_metric, cfg, metrics, epoch, task_idx, split)
    run.track({"acc" : fn("acc"), "f1" : fn("f1"), "loss" : fn("loss")}, context={"split": split, "task": task})

def log_metric(cfg, metrics, epoch, task_idx, split, metric_name):
    metrics_value = metrics[metric_name][split]
    return metrics_value[epoch, task_idx] if cfg.project == "miiii" else metrics_value[epoch]

def log_fn(cfg: Conf, ds, metrics):
    run = Run(experiment=cfg.project, system_tracking_interval=None)
    run["hparams"] = cfg.__dict__
    metrics = metrics_to_dict(metrics)

    # Log metrics for each epoch
    for epoch in tqdm(range(cfg.epochs)):
        for task_idx, task in enumerate(ds.info.tasks if cfg.project == "miiii" else range(1)):
            log_split(run, cfg, metrics, epoch, task, task_idx, "train")
            log_split(run, cfg, metrics, epoch, task, task_idx, "valid")

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

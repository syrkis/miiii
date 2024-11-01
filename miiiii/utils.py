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


# from aim import Run
from chex import dataclass
from typing import Literal


# %% constants
red = "#da3527"
blue = "#002fa7"


def check_nan(pytree, name):
    return jnp.array(tree.flatten(tree.map(lambda x: jnp.isnan(x).any(), pytree))[0]).any()


@dataclass
class Hyper:
    latent_dim: int = 128
    depth: int = 2
    heads: int = 4
    epochs: int = 1000
    lr: float = 1e-3
    l2: float = 0.1
    dropout: float = 0.1
    split: float = 0.5
    # seq_len: int  # if task is prose
    # vocab_size: int  # if task is prose


@dataclass
class Conf:
    hyper: Hyper
    task: str = "miiii"
    prime: int = 113

    # base: int
    # power: int = 2 # if we should use a different base.
    # task: str = "prime"  # "prose"
    # block: str = "vaswani"
    # causal: bool = False


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
def cfg_fn(kwargs, hyper_kwargs={}):
    cfg = Conf(**kwargs, hyper=Hyper(**hyper_kwargs))
    return cfg


def save_params(params, fname):
    path = os.path.join("data", fname)
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_params(fname):
    path = os.path.join("data", fname)
    with open(path, "rb") as file:
        return pickle.load(file)


# def track_metrics(metrics, ds, cfg):
#     run = Run(experiment="miiiii")
#     run["cfg"] = cfg.__dict__

#     for epoch in range(cfg.epochs):
#         for idx, task in enumerate(ds.info.tasks):
#             for split in ["train", "valid"]:
#                 to_log = {k: v[epoch][idx] for k, v in metrics[split].items()}
#                 run.track(to_log, epoch=epoch + 1, context={"task": task, "split": split})


def metrics_to_dict(metrics):
    train_metrics = dict(loss=metrics.train_loss.T, f1=metrics.train_f1.T)
    valid_metrics = dict(loss=metrics.valid_loss.T, f1=metrics.valid_f1.T)
    return dict(train=train_metrics, valid=valid_metrics)


def log_fn(cfg: Conf, ds, metrics):
    hyper = cfg.hyper
    cfg.__dict__.__delitem__("hyper")
    config = cfg.__dict__ | hyper.__dict__
    wandb.init(project=cfg.task, config=config, entity="syrkis", mode="offline")
    for epoch in range(hyper.epochs):
        for idx, task in enumerate(ds.info.tasks):  # type: ignore
            wandb.log(
                {
                    "train_loss": metrics.train.loss[epoch].item(),  # type: ignore
                    "train_f1": metrics.train.f1[epoch].item(),  # type: ignore
                    "train_acc": metrics.train.acc[epoch].item(),  # type: ignore
                    "valid_loss": metrics.valid.loss[epoch].item(),  # type: ignore
                    "valid_f1": metrics.valid.f1[epoch].item(),  # type: ignore
                    "valid_acc": metrics.valid.acc[epoch].item(),  # type: ignore
                },
                step=epoch,
            )

    # sync wandb
    wandb.finish()


def name_run_fn(cfg: Conf) -> str:
    """Create a descriptive run name from configuration.
    Format: task_ld{latent_dim}_de{depth}_he{heads}_lr{lr}_l2{l2}_dr{dropout}
    Example: miiii_ld128_de2_he4_lr1e-3_l20.1_dr0.1
    """
    return (
        f"{cfg.task}"
        f"_pm{cfg.prime}"
        f"_ld{cfg.hyper.latent_dim}"
        f"_de{cfg.hyper.depth}"
        f"_he{cfg.hyper.heads}"
        f"_lr{cfg.hyper.lr:g}"  # :g removes trailing zeros
        f"_l2{cfg.hyper.l2:g}"
        f"_dr{cfg.hyper.dropout:g}"
        f"_ep{cfg.hyper.epochs}"
    )


def init_db():
    """Initialize SQLite database with schema."""
    db_path = Path("metrics.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        with open("ddl.sql", "r") as f:
            conn.executescript(f.read())


def log_metrics_sql(metrics, cfg):
    """Log metrics to SQLite database, creating it if needed."""
    db_path = Path("metrics.db")
    ddl_path = Path("ddl.sql")

    # Initialize db if it doesn't exist
    if not db_path.exists():
        with sqlite3.connect(db_path) as conn:
            with open(ddl_path) as f:
                conn.executescript(f.read())

    with sqlite3.connect(db_path) as conn:
        # Ensure task exists in tasks table
        conn.execute(
            """
            INSERT OR IGNORE INTO tasks (task, description)
            VALUES (?, ?)
            """,
            (cfg.task, "Auto-inserted task"),  # You might want to add proper descriptions
        )

        # Log run config and get the run_id
        cursor = conn.execute(
            """
            INSERT INTO runs
            (task, prime, latent_dim, depth, heads, lr, l2, dropout, epochs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING run_id
            """,
            (
                cfg.task,
                cfg.prime,
                cfg.hyper.latent_dim,
                cfg.hyper.depth,
                cfg.hyper.heads,
                cfg.hyper.lr,
                cfg.hyper.l2,
                cfg.hyper.dropout,
                cfg.hyper.epochs,
            ),
        )
        run_id = cursor.fetchone()[0]

        # Prepare all metrics records
        records = []
        for split_name, split_data in [("train", metrics.train), ("valid", metrics.valid)]:
            for epoch in range(len(split_data.loss)):
                for task_id in range(split_data.loss.shape[1]):
                    task_name = f"{cfg.task}_{task_id}"  # Create unique task name for each task_id

                    # Ensure this task exists in tasks table
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO tasks (task, description)
                        VALUES (?, ?)
                        """,
                        (task_name, f"Task {task_id} for {cfg.task}"),
                    )

                    records.append(
                        (
                            run_id,
                            task_name,
                            epoch,
                            split_name,
                            float(split_data.loss[epoch, task_id]),
                            float(split_data.f1[epoch, task_id]),
                            float(split_data.acc[epoch, task_id]),
                        )
                    )

        # Bulk insert metrics
        conn.executemany(
            """
            INSERT INTO metrics
            (run_id, task, epoch, split, loss, f1, acc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

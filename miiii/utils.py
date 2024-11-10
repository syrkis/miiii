# miiii/miiii/utils.py
# miiii utils
# By: Noah Syrkis

# %% Imports
import os
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from chex import dataclass
from aim import Run, Image as AImage
from PIL import Image as PImage
import esch
import pickle
from oeis import oeis
import random
from omegaconf import DictConfig, ListConfig
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()


# Define your DigitalOcean Spaces endpoint
spaces_access_key_id = os.getenv('SPACES_ACCESS_KEY_ID')
spaces_secret_access_key = os.getenv('SPACES_SECRET_ACCESS_KEY')
spaces_endpoint = os.getenv('SPACES_ENDPOINT')
spaces_region = os.getenv('SPACES_REGION', 'ams3')  #


@dataclass
class Conf:
    p: int = 113
    project: str = "nanda"
    alpha: float = 0.98
    lamb: float = 2
    gamma: float = 2
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 1000
    lr: float = 1e-3
    l2: float = 1.0
    dropout: float = 0.5
    train_frac: float = 0.5


def sample_config(omegaconf: DictConfig | ListConfig) -> Conf:
    """
    Randomly samples from the configuration options provided in a DictConfig object
    and returns a Conf object with those selected hyperparameters.
    """
    return Conf(
        project="miiii",
        lr=random.choice(omegaconf.lr),
        l2=random.choice(omegaconf.l2),
        dropout=random.choice(omegaconf.dropout),
        heads=random.choice(omegaconf.heads),
        epochs=omegaconf.epochs,
        latent_dim=omegaconf.latent_dim,
    )


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


def metrics_to_dict(metrics):
    return {
        "loss": {
            "train": np.array(metrics.train.loss, dtype=np.float16),
            "valid": np.array(metrics.valid.loss, dtype=np.float16),
        },
        "f1": {
            "train": np.array(metrics.train.f1, dtype=np.float16),
            "valid": np.array(metrics.valid.f1, dtype=np.float16),
        },
        "acc": {
            "train": np.array(metrics.train.acc, dtype=np.float16),
            "valid": np.array(metrics.valid.acc, dtype=np.float16),
        },
    }


def log_split(run, cfg, metrics, epoch, task, task_idx, split):
    fn = partial(log_metric, cfg, metrics, epoch, task_idx, split)
    task = -1 if task == "prime" else int(task)
    run.track(
        {"acc": fn("acc"), "f1": fn("f1"), "loss": fn("loss")},
        context={"split": split, "task": task},
        step=epoch
    )


def log_metric(cfg, metrics, epoch, task_idx, split, metric_name):
    metrics_value = metrics[metric_name][split]
    return metrics_value[epoch, task_idx] if cfg.project == "miiii" else metrics_value[epoch]


def cfg_to_dirname(cfg: Conf) -> str:
    """
    Create a descriptive run name from configuration.
    Example: nanda_ld128_de1_he4_ep1000_lr1e-3_l21.0_dr0.5_tf0.5
    """
    param_order = [
        ('project', ''),
        ('latent_dim', 'ld'),
        ('depth', 'de'),
        ('heads', 'he'),
        ('epochs', 'ep'),
        ('lr', 'lr'),
        ('l2', 'l2'),
        ('dropout', 'dr'),
        ('train_frac', 'tf'),
        ('alpha', 'al'),
        ('lamb', 'la'),
        ('gamma', 'ga'),
        ('p', 'p'),
    ]

    name_parts = []
    for attr, abbrev in param_order:
        value = getattr(cfg, attr, None)

        if value is None or (isinstance(value, int) and value == 0):
            continue

        if isinstance(value, float):
            value = f"{value:.2e}" if attr in ['lr'] else f"{value:g}"

        name_parts.append(f"{abbrev}{value}")

    return "_".join(name_parts)


def log_fn(cfg, ds, state, metrics):
    run = Run(experiment=cfg.project, system_tracking_interval=None)

    run.set_artifacts_uri('s3://syrkis/')
    # make a dir data/artifacts/{run.hash}
    os.makedirs(f"data/artifacts/{run.hash}")


    # jnp.save(f"data/artifacts/{run.hash}/params.npy", state.params)
    # jnp.save(f"data/artifacts/{run.hash}/metrics.npy", metrics)
    with open(f"data/artifacts/{run.hash}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(f"data/artifacts/{run.hash}/params.pkl", "wb") as f:
        pickle.dump(state.params, f)

    run.log_artifact("data/artifacts/{run.hash}/metrics.pkl", name="state")
    run.log_artifact("data/artifacts/{run.hash}/params.pkl", name="state")

    run["hparams"] = {k: v for k, v in cfg.__dict__.items() if k not in ["project", "debug", "prime"]}
    run["dataset"] = {"prime": cfg.p, "project": cfg.project}

    metrics_dict = metrics_to_dict(metrics)
    tasks = [p for p in oeis["A000040"][1:cfg.p] if p < cfg.p]

    log_steps = 1000
    for epoch in range(0, cfg.epochs, max(1, cfg.epochs // log_steps)):
        for task_idx, task in enumerate(tasks if cfg.project == "miiii" else range(1)):
            log_split(run, cfg, metrics_dict, epoch, task, task_idx, "train")
            log_split(run, cfg, metrics_dict, epoch, task, task_idx, "valid")

    p = esch.plot(state.params.embeds.tok_emb)
    run.track(AImage(PImage.open(p.png)), name="tok_emb", step=cfg.epochs)  # type: ignore

    run.close()


# Initialize the S3 client for DigitalOcean Spaces
s3_client = boto3.client(
    's3',
    region_name=spaces_region,
    endpoint_url=spaces_endpoint,
    aws_access_key_id=spaces_access_key_id,
    aws_secret_access_key=spaces_secret_access_key
)

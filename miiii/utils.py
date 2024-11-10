# miiii/miiii/utils.py
# miiii utils
# By: Noah Syrkis

# %% Imports
import os
import jax.numpy as jnp
import numpy as np
from functools import partial
from chex import dataclass
from aim import Run, Image as AImage, Repo
from PIL import Image as PImage
import esch
import pickle
from oeis import oeis
import random
from omegaconf import DictConfig, ListConfig


# Define your DigitalOcean Spaces endpoint
spaces_access_key_id = os.getenv("SPACES_ACCESS_KEY_ID")
spaces_secret_access_key = os.getenv("SPACES_SECRET_ACCESS_KEY")
spaces_endpoint = os.getenv("SPACES_ENDPOINT")
spaces_region = os.getenv("SPACES_REGION", "ams3")  #


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
        {"acc": fn("acc"), "f1": fn("f1"), "loss": fn("loss")}, context={"split": split, "task": task}, step=epoch
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
        ("project", ""),
        ("latent_dim", "ld"),
        ("depth", "de"),
        ("heads", "he"),
        ("epochs", "ep"),
        ("lr", "lr"),
        ("l2", "l2"),
        ("dropout", "dr"),
        ("train_frac", "tf"),
        ("alpha", "al"),
        ("lamb", "la"),
        ("gamma", "ga"),
        ("p", "p"),
    ]

    name_parts = []
    for attr, abbrev in param_order:
        value = getattr(cfg, attr, None)

        if value is None or (isinstance(value, int) and value == 0):
            continue

        if isinstance(value, float):
            value = f"{value:.2e}" if attr in ["lr"] else f"{value:g}"

        name_parts.append(f"{abbrev}{value}")

    return "_".join(name_parts)


def log_fn(cfg, ds, state, metrics, acts):
    run = Run(experiment=cfg.project, system_tracking_interval=None)
    run.set_artifacts_uri("s3://syrkis/")
    grand_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_hash_dir = os.path.join(grand_parent, "data/artifacts", run.hash)
    os.makedirs(run_hash_dir, exist_ok=True)

    with open(f"{run_hash_dir}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(f"{run_hash_dir}/state.pkl", "wb") as f:
        pickle.dump(state, f)
    with open(f"{run_hash_dir}/acts.pkl", "wb") as f:
        pickle.dump(acts, f)

    run.log_artifact(f"{run_hash_dir}/metrics.pkl", name="metrics.pkl", block=True)
    run.log_artifact(f"{run_hash_dir}/state.pkl", name="state.pkl", block=True)
    run.log_artifact(f"{run_hash_dir}/acts.pkl", name="acts.pkl", block=True)

    run["hparams"] = {k: v for k, v in cfg.__dict__.items() if k not in ["project", "debug", "prime"]}
    run["dataset"] = {"prime": cfg.p, "project": cfg.project}

    metrics_dict = metrics_to_dict(metrics)
    tasks = [p for p in oeis["A000040"][1 : cfg.p] if p < cfg.p]

    log_steps = 1000
    for epoch in range(0, cfg.epochs, max(1, cfg.epochs // log_steps)):
        for task_idx, task in enumerate(tasks if cfg.project == "miiii" else range(1)):
            log_split(run, cfg, metrics_dict, epoch, task, task_idx, "train")
            log_split(run, cfg, metrics_dict, epoch, task, task_idx, "valid")

    p = esch.plot(state.params.embeds.tok_emb)
    run.track(AImage(PImage.open(p.png)), name="tok_emb", step=cfg.epochs)  # type: ignore

    run.close()


def get_metrics_and_params(hash):
    hash_run_dir = os.path.join(os.getcwd(), "data/artifacts", hash)
    os.makedirs(hash_run_dir, exist_ok=True)
    repo = Repo("aim://localhost:53800")  # make sure this is running
    run = repo.get_run(hash)
    outs = {"state": None, "metrics": None, "acts": None}

    for thing in outs.keys():
        file_path = os.path.join(hash_run_dir, f"{thing}.pkl")

        # Check if the file already exists before downloading
        if not os.path.exists(file_path):
            run.artifacts[f"{thing}.pkl"].download(hash_run_dir)  # type: ignore

        # Load the file content
        with open(file_path, "rb") as f:
            outs[thing] = pickle.load(f)

    return outs["state"], (outs["metrics"], outs["acts"])

# miiii/miiii/utils.py
# miiii utils
# By: Noah Syrkis

# %% Imports
import argparse
import os
import pickle
import sys
from dataclasses import field
from functools import partial

# import esch
import jax.numpy as jnp
import numpy as np
from aim import Repo, Run

from chex import dataclass
from jax import Array
from oeis import oeis


# %% Types
@dataclass
class Activation:
    wei: Array
    ffwd: Array = field(default_factory=lambda: jnp.array([]))
    logits: Array = field(default_factory=lambda: jnp.array([]))


@dataclass
class Split:
    loss: Array
    f1: Array
    acc: Array


@dataclass
class Metrics:
    train: Split
    valid: Split


# %% Data classes
@dataclass
class Feedforward:
    w_in: Array
    w_out: Array


@dataclass
class Attention:
    q: Array
    k: Array
    v: Array
    o: Array


@dataclass
class Embedding:
    tok_emb: Array
    pos_emb: Array


@dataclass
class Params:
    embeds: Embedding
    ffwd: Feedforward
    attn: Attention
    unbeds: Array  # should be a linear layer ?


@dataclass
class State:
    params: Params
    opt_state: Params
    emas: Params


def parse_args():
    parser = argparse.ArgumentParser(description="Run model with specified hyperparameters.")

    # Define all hyperparameters as optional arguments with defaults from Conf class
    parser.add_argument("--project", type=str, help="Project name")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension size")
    parser.add_argument("--depth", type=int, help="Depth of the model")
    parser.add_argument("--heads", type=int, help="Number of attention heads")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--task", type=str, help="Task")
    parser.add_argument("--l2", type=float, help="L2 regularization")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--train_frac", type=float, help="Fraction of data used for training")
    parser.add_argument("--alpha", type=float, help="Alpha value for optimization")
    parser.add_argument("--lamb", type=float, help="Lambda value for regularization")
    parser.add_argument("--gamma", type=float, help="Gamma value for optimization")
    parser.add_argument("--p", type=int, help="Prime number for data configuration")

    return parser.parse_args()


@dataclass
class Conf:
    p: int = 113
    project: str = "miiii"
    alpha: float = 0.98
    lamb: float = 2
    task: str = "multi"  # binary, multi
    gamma: float = 2
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 10000
    lr: float = 1e-4
    l2: float = 1.0
    dropout: float = 0.5
    train_frac: float = 0.5


def create_cfg(**kwargs) -> Conf:
    """
    Create a configuration object from parsed command-line arguments.
    """
    if "ipykernel" not in sys.argv[0]:
        cli_args = parse_args()
        for key in kwargs:
            assert (
                getattr(cli_args, key, None) is None
            ), f"Duplicate argument: {key}"  # asset that everything in kwargs has none value in cliargs
        # merge cli_args and kwargs
        kwargs = {**cli_args.__dict__, **kwargs}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        cfg = Conf(**kwargs)
        return cfg
    else:
        return Conf(**kwargs)


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
    run = Run(experiment=cfg.project, system_tracking_interval=None, repo="aim://localhost:53800")
    run.set_artifacts_uri("s3://syrkis/")
    grand_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_hash_dir = os.path.join(grand_parent, "data/artifacts", run.hash)
    print(cfg)
    print(run.hash)
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

    # p = esch.plot(state.params.embeds.tok_emb)
    # run.track(AImage(PImage.open(p.png)), name="tok_emb", step=cfg.epochs)  # type: ignore

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

    state: State = outs["state"]  # type: ignore
    metrics: Metrics = outs["metrics"]  # type: ignore
    acts: Activation = outs["acts"]  # type: ignore
    return state, (metrics, acts)


def construct_cfg_from_hash(hash: str) -> Conf:
    """
    Constructs the configuration object from a specific run identified by its hash.
    """
    # Define the path to the hash directory and access the aim repository
    repo = Repo("aim://localhost:53800")  # ensure aim server is running
    run = repo.get_run(hash)

    if run is None:
        raise ValueError(f"No run associated with hash: {hash}")

    # Retrieve hyperparameters stored in the run
    hparams = run["hparams"]
    dataset = run["dataset"]

    # Create and return a Conf instance using the retrieved parameters from the run
    return Conf(
        project=dataset.get("project", "miiii"),  # type: ignore
        p=run["dataset"].get("prime", 113),  # assuming it stores the prime as well  # type: ignore
        alpha=hparams.get("alpha", 0.98),  # type: ignore
        lamb=hparams.get("lamb", 2),  # type: ignore
        gamma=hparams.get("gamma", 2),  # type: ignore
        latent_dim=hparams.get("latent_dim", 128),  # type: ignore
        depth=hparams.get("depth", 1),  # type: ignore
        heads=hparams.get("heads", 4),  # type: ignore
        epochs=hparams.get("epochs", 1000),  # type: ignore
        lr=hparams.get("lr", 3e-4),  # type: ignore
        l2=hparams.get("l2", 1.0),  # type: ignore
        dropout=hparams.get("dropout", 0.5),  # type: ignore
        train_frac=hparams.get("train_frac", 0.5),  # type: ignore
    )

# miiii/miiii/utils.py
# miiii utils
# By: Noah Syrkis

# %% Imports
import argparse
import os
import sys
import pickle
from dataclasses import field
from functools import partial
import yaml

# import esch
import jax.numpy as jnp
import numpy as np
from aim import Repo, Run
from chex import dataclass
from jaxtyping import Array
from oeis import oeis
from tqdm import tqdm


# %% Types
@dataclass
class Activation:
    wei: Array
    ffwd: Array = field(default_factory=lambda: jnp.array([]))
    logits: Array = field(default_factory=lambda: jnp.array([]))


@dataclass
class Split:
    loss: Array
    # f1: Array
    acc: Array


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
class Metrics:
    train: Split
    valid: Split


@dataclass
class Scope:
    # logit_freqs: Array
    grad_norms: Params | None
    neuron_freqs: Array


@dataclass
class State:
    params: Params
    opt_state: Params | None = None
    emas: Params | None = None


@dataclass
class Conf:
    # alpha: float = 0.98
    p: int = 113
    lamb: float = 2
    latent_dim: int = 128
    depth: int = 1
    heads: int = 4
    epochs: int = 20000
    lr: float = 3e-4  # i just usually do this.
    l2: float = 1.0
    dropout: float = 0.5
    train_frac: float = 0.5
    mask: bool = False  # weather to mask first four tasks
    shuffle: bool = False  # weather to shuffle the y labels


def cfg_fn() -> Conf:
    """Create a configuration object from parsed command-line arguments."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)["default"]
    return Conf(**cfg)


def arg_fn():
    parser = argparse.ArgumentParser(description="Run model with specified hyperparameters.")
    parser.add_argument("--runs", type=int, help="Number of trials to run", default=10)
    parser.add_argument("--tick", type=int, help="Number of trials to run", default=100)  # how often to scope
    parser.add_argument(
        "--task", type=str, help="Which task to train on", default="miiii"
    )  # nanda or baseline or miiii
    parser.add_argument(
        "--mods", type=str, help="Weather to test divisibility or remainders", default="remainder"
    )  # remainder or divisibility
    parser.add_argument("--mask", type=bool, help="should i mask the first four tasks?")
    if "ipykernel" not in sys.argv[0]:
        return parser.parse_args()
    else:
        return parser.parse_args(["--runs", "1", "--task", "miiii", "--mods", "remainder", "--mask", "True"])


def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


def metrics_to_dict(metrics):
    return {
        "loss": {"train": np.array(metrics.train.loss), "valid": np.array(metrics.valid.loss)},
        # "f1": {"train": np.array(metrics.train.f1), "valid": np.array(metrics.valid.f1)},
        "acc": {"train": np.array(metrics.train.acc), "valid": np.array(metrics.valid.acc)},
    }


def log_split(run, cfg, metrics, epoch, factor, task_idx, split, task_type, task_span):
    fn = partial(log_metric, cfg, metrics, epoch, task_idx, split, task_type, task_span)
    factor = -1 if factor == "prime" else int(factor)
    run.track(
        {"acc": fn("acc"), "loss": fn("loss")},
        context={"split": split, "factor": factor, "task_type": task_type, "task_span": task_span},
        step=epoch,
    )


def log_metric(cfg, metrics, epoch, factor_idx, split, task_type, task_span, metric_name):
    metrics_value = metrics[metric_name][split]
    return metrics_value[epoch, factor_idx] if task_span == "factors" else metrics_value[epoch]


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


def log_fn(state, metrics, scope, cfg, ds, task):
    # hash cfg
    run = Run(
        experiment="miiii", system_tracking_interval=None, capture_terminal_logs=False
    )  # , repo="aim://localhost:53800")
    run.set_artifacts_uri("s3://syrkis/")
    grand_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_hash_dir = os.path.join(grand_parent, "data/artifacts", run.hash)
    os.makedirs(run_hash_dir, exist_ok=True)

    with open(f"{run_hash_dir}/metrics_{task.type}_{task.span}.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(f"{run_hash_dir}/state_{task.type}_{task.span}.pkl", "wb") as f:
        pickle.dump(state, f)
    with open(f"{run_hash_dir}/scope_{task.type}_{task.span}.pkl", "wb") as f:
        pickle.dump(scope, f)

    run.log_artifact(
        f"{run_hash_dir}/metrics_{task.type}_{task.span}.pkl",
        name=f"metrics_{task.type}_{task.span}.pkl",
        block=True,
    )
    run.log_artifact(
        f"{run_hash_dir}/state_{task.type}_{task.span}.pkl", name=f"state_{task.type}_{task.span}.pkl", block=True
    )
    run.log_artifact(
        f"{run_hash_dir}/scope_{task.type}_{task.span}.pkl", name=f"scope_{task.type}_{task.span}.pkl", block=True
    )

    run["hparams"] = cfg.__dict__

    metrics_dict = metrics_to_dict(metrics)
    factors = [p for p in oeis["A000040"][1 : cfg.p] if p < cfg.p]

    log_steps = 1000
    for epoch in tqdm(range(0, cfg.epochs, max(1, cfg.epochs // log_steps))):
        for factor_idx, factor in enumerate(factors if task.span == "factors" else range(1)):
            log_split(run, cfg, metrics_dict, epoch, factor, factor_idx, "train", task.type, task.span)
            log_split(run, cfg, metrics_dict, epoch, factor, factor_idx, "valid", task.type, task.span)

    # p = esch.plot(state.params.embeds.tok_emb)
    # run.track(AImage(PImage.open(p.png)), name="tok_emb", step=cfg.epochs)  # type: ignore

    run.close()


def get_metrics_and_params(hash, task_span="factors"):
    hash_run_dir = os.path.join(os.getcwd(), "data/artifacts", hash)
    os.makedirs(hash_run_dir, exist_ok=True)
    repo = Repo("aim://localhost:53800")  # make sure this is running
    run = repo.get_run(hash)

    outs_list = []
    for task_type, task_span in [("remainder", task_span)]:
        outs = {"state": None, "metrics": None, "scope": None}
        for thing in outs.keys():
            file_path = os.path.join(hash_run_dir, f"{thing}_{task_type}_{task_span}.pkl")

            # Check if the file already exists before downloading
            if not os.path.exists(file_path):
                try:
                    run.artifacts[f"{thing}_{task_type}_{task_span}.pkl"].download(hash_run_dir)  # type: ignore
                except:  # noqa
                    print(f"Could not download {thing}_{task_type}_{task_span}.pkl")

            # Load the file content
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    outs[thing] = pickle.load(f)
            else:
                continue

        state: State = outs["state"]  # type: ignore
        metrics: Metrics = outs["metrics"]  # type: ignore
        scope: Scope = outs["scope"]  # type: ignore
        cfg = construct_cfg_from_hash(hash)
        outs_list.append((state, metrics, scope, cfg))

    return outs_list[0]  # , outs_list[1]


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
    # dataset = run["dataset"]

    # Create and return a Conf instance using the retrieved parameters from the run
    return Conf(
        # project=dataset.get("project", "miiii"),  # type: ignore
        # p=run["dataset"].get("prime", 113),  # assuming it stores the prime as well  # type: ignore
        # alpha=hparams.get("alpha", 0.98),  # type: ignore
        # gamma=hparams.get("gamma", 2),  # type: ignore
        lamb=hparams.get("lamb", 2),  # type: ignore
        latent_dim=hparams.get("latent_dim", 128),  # type: ignore
        depth=hparams.get("depth", 1),  # type: ignore
        heads=hparams.get("heads", 4),  # type: ignore
        epochs=hparams.get("epochs", 1000),  # type: ignore
        lr=hparams.get("lr", 3e-4),  # type: ignore
        l2=hparams.get("l2", 1.0),  # type: ignore
        dropout=hparams.get("dropout", 0.5),  # type: ignore
        train_frac=hparams.get("train_frac", 0.5),  # type: ignore
    )


def fourier_basis(p):  # TODO This is a bit wrong
    freqs = jnp.arange(1, p // 2 + 1)[:, None]
    phase = 2 * jnp.pi * freqs * jnp.arange(p) / p
    F = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)])
    return F / jnp.linalg.norm(F, axis=1, keepdims=True)

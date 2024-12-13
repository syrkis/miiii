# miiii/miiii/utils.py
# miiii utils
# By: Noah Syrkis

# %% Imports
import argparse
import os
import pickle
import sys
from dataclasses import field
from typing import Any, Dict

# import esch
import jax.numpy as jnp
import numpy as np
import yaml
from aim import Repo, Run
from chex import dataclass
from jaxtyping import Array


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
    alpha: float = 0.98


def cfg_fn(search_space=False):
    """Create a configuration object from parsed command-line arguments."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return Conf(**cfg['default']) if not search_space else cfg


def arg_fn():
    parser = argparse.ArgumentParser(description="Run model with specified hyperparameters.")
    parser.add_argument("--runs", type=int, help="Number of trials to run", default=1)
    parser.add_argument("--tick", type=int, help="Number of trials to run", default=100)  # how often to scope
    parser.add_argument("--task", type=str, help="Which task to train on", default="miiii")
    parser.add_argument("--mods", type=str, help="Weather to test divisibility or remainders", default="remainder")
    parser.add_argument("--mask", type=bool, help="should i mask the first four tasks?")
    if "ipykernel" not in sys.argv[0]:
        return parser.parse_args()
    return parser.parse_args(["--runs", "1", "--task", "miiii", "--mods", "remainder", "--mask", "True"])


def metrics_to_dict(metrics):
    return {
        "loss": {"train": np.array(metrics.train.loss), "valid": np.array(metrics.valid.loss)},
        "acc": {"train": np.array(metrics.train.acc), "valid": np.array(metrics.valid.acc)},
    }


def init_log(cfg, arg):
    run = Run(experiment="miiii", system_tracking_interval=None, capture_terminal_logs=False)
    grand_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_hash_dir = os.path.join(grand_parent, "data/artifacts", run.hash)
    os.makedirs(run_hash_dir, exist_ok=True)
    run.set_artifacts_uri("s3://syrkis/")
    run["hparams"] = cfg.__dict__
    run["args"] = arg.__dict__
    return run, run_hash_dir


def store_artifact(run, name, obj, run_hash_dir):
    with open(f"{run_hash_dir}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    run.log_artifact(f"{run_hash_dir}/{name}.pkl", name=f"{name}.pkl", block=True)


def log_fn(cfg, arg, ds, state, metrics):
    run, run_hash_dir = init_log(cfg, arg)
    store_artifact(run, "state", state, run_hash_dir)
    metrics = metrics_to_dict(metrics)
    for tick in range(arg.tick):
        for idx, prime in enumerate(ds.primes.tolist()):
            for split in ["train", "valid"]:
                run.track(
                    {
                        "acc": metrics["acc"][split][(tick, idx) if arg.task == "miiii" else tick].item(),
                        "loss": metrics["loss"][split][(tick, idx) if arg.task == "miiii" else tick].item(),
                    },
                    context={"split": split, "prime": prime},
                    step=tick * (cfg.epochs // arg.tick),
                )


def get_metrics_and_params(hash, task_span="factors"):
    hash_run_dir = os.path.join(os.getcwd(), "data/artifacts", hash)
    os.makedirs(hash_run_dir, exist_ok=True)
    repo = Repo("aim://localhost:53800")  # make sure this is running
    run = repo.get_run(hash)
    file_path = os.path.join(hash_run_dir, "state.pkl")
    run.artifacts["state.pkl"].download(hash_run_dir)  # type: ignore
    with open(file_path, "rb") as f:
        state = pickle.load(f)
    return state


def construct_cfg_from_hash(hash: str) -> Conf:
    repo = Repo("aim://localhost:53800")  # ensure aim server is running
    cfg: Dict[str, Any] = repo.get_run(hash)["hparams"]  # type: ignore
    return Conf(**cfg)

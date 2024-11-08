# %% utils.py
#   miiii utils
# by: Noah Syrkis

# %% imports
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from chex import dataclass
from aim import Run, Image as AImage
from PIL import Image as PImage
import esch
from oeis import oeis
import random  # should this be determinisitc with jax?
from omegaconf import DictConfig, ListConfig


@dataclass
class Conf:
    p: int = 113  # @nanda2023
    project: str = "nanda"
    alpha: float = 0.98  # not sure what this does (grokfast)
    lamb: float = 2  # set to 0 for no filter (grokfast)
    gamma: float = 2  # grokfast
    latent_dim: int = 128  # @nanda2023
    depth: int = 1  # @nanda2023
    heads: int = 4  # @nanda2023
    epochs: int = 1000
    lr: float = 1e-3  # @nanda2023
    l2: float = 1.0  # @nanda2023
    dropout: float = 0.5  # @nanda2023
    train_frac: float = 0.5  # @nanda2023

def sample_config(omegaconf: DictConfig | ListConfig) -> Conf:
    """
    Randomly samples from the configuration options provided in a DictConfig object
    and returns a Conf object with those selected hyperparameters.
    """
    return Conf(
        # Assuming the project is a fixed string, not part of the random search
        project="miiii",

        # Randomly sample each hyperparameter from the provided configuration space
        lr=random.choice(omegaconf.lr),
        l2=random.choice(omegaconf.l2),
        dropout=random.choice(omegaconf.dropout),
        heads=random.choice(omegaconf.heads),

        # The following are not parameterized in the configuration and remain static
        epochs=omegaconf.epochs,      # assuming this is meant to be the number of iterations
        latent_dim=omegaconf.latent_dim,

        # If other hyperparameters like `alpha`, `lamb`, `p`, `depth`, `train_frac`, or `debug`
        # are also part of the configuration, decide whether to use defaults or include them in the search.
    )



def digit_fn(n, base):
    return jnp.ceil(jnp.log(n + 1) / jnp.log(base)).astype(jnp.int32)


# %% functions
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


def log_fn(cfg: Conf, ds, state, metrics):
    run = Run(experiment=cfg.project, system_tracking_interval=None)
    run["params"] = cfg.__dict__
    metrics = metrics_to_dict(metrics)
    tasks = [p for p in oeis["A000040"][1 : cfg.p] if p < cfg.p]

    # Log metrics for each epoch
    log_steps = 1000
    for epoch in tqdm(range(0, cfg.epochs, cfg.epochs // log_steps)):
        for task_idx, task in enumerate(tasks if cfg.project == "miiii" else range(1)):
            log_split(run, cfg, metrics, epoch, task, task_idx, "train")
            log_split(run, cfg, metrics, epoch, task, task_idx, "valid")

    p = esch.plot(state.params.embeds.tok_emb)
    run.track(AImage(PImage.open(p.png)), name="tok_emb", step=cfg.epochs)  # type: ignore

    run.close()


def name_run_fn(cfg: Conf) -> str:
    """Create a descriptive run name from configuration.
    Format: task_ld{latent_dim}_de{depth}_he{heads}_lr{lr}_l2{l2}_dr{dropout}
    Example: miiii_ld128_de2_he4_lr1e-3_l20.1_dr0.1
    """
    return (
        # f"{cfg.task}"
        f"_pm{cfg.p}"
        f"_ld{cfg.latent_dim}"
        f"_de{cfg.depth}"
        f"_he{cfg.heads}"
        f"_lr{cfg.lr:g}"  # :g removes trailing zeros
        f"_l2{cfg.l2:g}"
        f"_dr{cfg.dropout:g}"
        f"_ep{cfg.epochs}"
    )

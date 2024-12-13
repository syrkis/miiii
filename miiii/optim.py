# optim.py
import optuna
import yaml
from typing import Dict, Any
from jax import random

from miiii.utils import Conf
from miiii.train import train_fn
from miiii.tasks import Dataset


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_trial_config(trial: optuna.Trial, config: Dict[str, Any]) -> Conf:
    """Create a configuration from an Optuna trial."""
    search_space = config["search_space"]
    default = config["default"]

    params = {
        "depth": trial.suggest_int("depth", *search_space["depth"]),
        "heads": trial.suggest_int("heads", *search_space["heads"]),
        "latent_dim": trial.suggest_int("latent_dim", *search_space["latent_dim"]),
        "lr": trial.suggest_float("lr", *search_space["lr"], log=True),
        "l2": trial.suggest_float("l2", *search_space["l2"], log=True),
        "lamb": trial.suggest_float("lamb", *search_space["lamb"], log=True),
        "dropout": trial.suggest_float("dropout", *search_space["dropout"]),
    }

    # Merge with default config
    full_params = {**default, **params}
    return Conf(**full_params)


def objective(trial: optuna.Trial, dataset: Dataset):
    """Optuna objective function."""
    config = load_config()
    cfg = create_trial_config(trial, config)

    # Train model
    rng = random.PRNGKey(0)
    state, metrics = train_fn(rng, cfg, dataset)

    # Return final validation accuracy
    return float(metrics.valid.acc[-1].mean())


def optimize(dataset: Dataset, task, n_trials: int = 100):
    """Run hyperparameter optimization."""
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=0)
    )

    study.optimize(
        lambda trial: objective(trial, dataset),
        n_trials=n_trials,
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"Best accuracy: {best_value:.4f}")
    print("Best hyperparameters:", best_params)

    return best_params, best_value

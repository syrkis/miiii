# optim.py
#   miiiiiiiii hyperparameter optimization
# by: Noah Syrkis

# Imports
import optuna
import yaml
from miiii.utils import Conf, log_fn
from typing import Dict, Any
from jax import random
from functools import partial
from miiii.train import train_fn
from miiii.tasks import task_fn


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_trial_config(trial: optuna.Trial, config: Dict[str, Any]) -> Conf:
    trial_params = config["default"].copy()
    search_space = config.get("search_space", {})

    for param, values in search_space.items():
        trial_params[param] = trial.suggest_categorical(param, values)

    return Conf(**trial_params)


def objective(arg, trial: optuna.Trial) -> float:
    cfg = create_trial_config(trial, load_config())
    rng = random.PRNGKey(trial.number)
    ds = task_fn(rng, cfg, arg)
    state, (metrics, loss) = train_fn(rng, cfg, arg, ds)
    log_fn(cfg, arg, ds, state, metrics)
    return metrics.valid.acc[-1].mean()


def run_study(arg):
    study = optuna.create_study(
        direction="maximize", study_name="miiii_optimization", sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(partial(objective, arg), n_trials=arg.runs)

    # Print summary
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return study

# %% main.py
#   miiii notebook
# by: Noah Syrkis

# %% Imports
import miiii as mi

# import esch
from jax import random
from functools import partial
import optuna
import yaml


def evaluate_config(config, rng) -> float:
    """Evaluate a single configuration"""
    cfg = mi.utils.create_cfg(**config)
    ds, task = mi.tasks.task_fn(rng, cfg, "remainder", "factors")  # only task i have
    state, (metrics, acts) = mi.train.train(rng, cfg, ds, task)
    mi.utils.log_fn(state, metrics, acts, cfg, ds, task)
    return metrics.valid.loss[-10:].mean().item()  # mean over last 10 epochs


def objective_fn(trial, base_config, search_space, rng):
    config = base_config.copy()

    for param, values in search_space.items():
        # trial_suggest_fn = trial.suggest_float if isinstance(values[0], float) else trial.suggest_int
        # config[param] = trial_suggest_fn(param, min(values), max(values))
        config[param] = trial.suggest_categorical(param, values)

    return evaluate_config(config, rng)


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Split into base config and search space
    search_space = config.pop("search_space")
    study = optuna.create_study(direction="minimize")
    rng = random.PRNGKey(0)

    # Run optuna
    objective = partial(objective_fn, base_config=config, search_space=search_space, rng=rng)
    args = mi.utils.parse_args()
    study.optimize(objective, n_trials=args.runs)

    # Print results
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()

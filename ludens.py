# %% ludens.py
#   ludens notebook
# by: Noah Syrkis


# %% Imports
from aim import Repo
import os
import pickle


# %%
hash = "09411cac9a174bb3a95db629"


def get_metrics_and_params(hash):
    hash_run_dir = os.path.join(os.getcwd(), "data/artifacts", hash)
    os.makedirs(hash_run_dir, exist_ok=True)
    repo = Repo("aim://localhost:53800")
    run = repo.get_run(hash)
    outs = {"state": None, "metrics": None, "acts": None}
    for thing in ["matrics", "state", "acts"]:
        run.artifacts[f"{thing}.pkl"].download(os.path.join(hash_run_dir))  # type: ignore
        with open(os.path.join(hash_run_dir, f"{thing}.pkl"), "rb") as f:
            outs[thing] = pickle.load(f)
    return outs["state"], (outs["metrics"], outs["acts"])


# %%
state, (metrics, acts) = get_metrics_and_params(hash)

# %% ludens.py
#   ludens notebook
# by: Noah Syrkis


# %% Imports
import miiii as mi
from aim import Repo
import os



hash = "09411cac9a174bb3a95db629"
hash_run_dir = os.path.join(os.getcwd(), "data/artifacts", hash)
os.makedirs(hash_run_dir, exist_ok=True)
repo = Repo('aim://localhost:53800')
run = repo.get_run(hash)
run.artifacts["metrics.pkl"].download(os.path.join(hash_run_dir, "metrics.pkl"))  # type: ignore
run.artifacts["params.pkl"].download(os.path.join(hash_run_dir, "params.pkl"))  # type: ignore

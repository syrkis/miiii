# ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
from jax import random, tree, lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# %% Initialize
cfg, (rng, key) = mi.get_conf(), random.split(random.PRNGKey(seed := 0))
train_data, valid_data, tasks = mi.data_fn(cfg.n, cfg.base, mi.base_ns, rng)
params = mi.init_fn(key, cfg, *train_data)


# %% Functions
apply_fn = mi.make_apply_fn(mi.vaswani_fn)
args = (apply_fn, params, cfg, mi.alpha_fn, train_data, valid_data)
train_fn, state = mi.init_train(*args)


# %% Train
state, metrics = train_fn(cfg.epochs, rng, state)

# %% Evaluate
fig, ax = plt.subplots()
# colors = plt.cm.get_cmap("Greys", len(tasks))  + red
colors = sns.color_palette("Greys", len(tasks) - 1) + ["red"]
# ax.plot(metrics["train_loss"], label=tasks, color=colors)
sns.lineplot(
    data=pd.DataFrame(metrics["train_loss"], columns=tasks), ax=ax, palette=colors
)

# %%
# colors is white to black in len(tasks) steps
colors

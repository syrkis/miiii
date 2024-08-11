# ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns


# %% Initialize
cfg, (rng, key) = mi.utils.get_conf(), random.split(random.PRNGKey(seed := 0))
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, rng)
params = mi.param.init_fn(key, cfg, ds.train.x, ds.train.y)


# %% Functions
apply_fn = mi.model.make_apply_fn(mi.model.vaswani_fn)
train_fn, state = mi.train.init_train(apply_fn, params, cfg, mi.utils.alpha_fn, ds)


# %% Train
state, metrics = train_fn(cfg.epochs, rng, state)

# %% Evaluate
fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
colors = sns.color_palette("Greys", len(ds.info.tasks) - 1) + [mi.utils.red]
sns.lineplot(data=metrics["valid_loss"], ax=ax, palette=colors)  # TODO: confirm order

# %%

# %%
# fig, ax = plt.subplots(figsize=(100, 4), dpi=100)
# sns.heatmap(train_data[1].T, ax=ax, cmap="Greys", cbar=False)
# train_data[1].shape

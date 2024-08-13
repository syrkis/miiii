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
fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
colors = sns.color_palette("Greys", len(ds.info.tasks) - 1) + [mi.utils.blue]
ax.set_title("Training Focal Loss")
ax.set_xlabel("Epoch")
# ax.set_ylabel("Focal Loss")
sns.lineplot(
    data=metrics["train_loss"],
    ax=ax,
    palette=colors,
    legend=False,
)  # TODO: confirm order
# set ylim to 0, 1 and only show 0 and 1
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
# set legend to task names
plt.tight_layout()
plt.savefig("paper/figs/training_loss_curves.svg")

# %%

# %%
# fig, ax = plt.subplots(figsize=(100, 4), dpi=100)
# sns.heatmap(train_data[1].T, ax=ax, cmap="Greys", cbar=False)
# train_data[1].shape

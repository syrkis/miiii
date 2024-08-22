# ludens.py
#   miiiii notebook
# by: Noah Syrkis


# %% Imports
import miiiii as mi
from jax import random
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import seaborn as sns

# %%
rng = random.PRNGKey(0)
X = jnp.eye(100)
Y = jnp.fft.fftn(X).astype(jnp.int8)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(X)
axes[1].imshow(Y)
# %% Initialize
cfg, (rng, key) = mi.utils.get_conf(), random.split(random.PRNGKey(seed := 0))
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, rng)
params = mi.param.init_fn(key, cfg, ds.train.x, ds.train.y)


# %% Functions
apply_fn = mi.model.make_apply_fn(mi.model.vaswani_fn)
train_fn, state = mi.train.init_train(apply_fn, params, cfg, mi.utils.alpha_fn, ds)
state, metrics = train_fn(cfg.epochs, rng, state)


# %% Train
mi.plots.syrkis_plot(metrics["train_loss"], cfg, "Train Focal Loss", ds)

# %% Valid
mi.plots.syrkis_plot(metrics["valid_loss"], cfg, "Valid Focal Loss", ds)

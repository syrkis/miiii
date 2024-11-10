# %% miiiii.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiii as mi
import sys
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    args = mi.utils.parse_args()
    cfg = mi.utils.create_cfg(args)


else:
    cfg = mi.utils.Conf(project="nanda", p=37, train_frac=0.8, lr=3e-4, epochs=1000, latent_dim=32)

rng, *keys = random.split(random.PRNGKey(0), 3)
ds = mi.tasks.task_fn(keys[0], cfg)
ds.train[0].shape
# %%
state, (metrics, acts) = mi.train.train(keys[1], cfg, ds)
# x = jnp.stack((ds.train[0], ds.valid[0]))[jnp.argsort(ds.idxs)]
# y = jnp.stack((ds.train[1], ds.valid[1]))[jnp.argsort(ds.idxs)]
# acts =

# %%
mi.utils.log_fn(cfg, ds, state, metrics, acts)

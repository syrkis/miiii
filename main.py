# %% main.py
#   miiiii notebook
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random, tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
from oeis import A000040 as primes

# %% Exploring and plotting the data
cfg, rng = mi.utils.get_conf(), random.PRNGKey(seed := 0)
rng, key = random.split(rng)
ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns, key)
params = mi.param.init_fn(key, cfg, ds.train.x, ds.train.y)
apply = mi.model.make_apply_fn(mi.model.vaswani_fn)
train, state = mi.train.init_train(apply, params, cfg, mi.utils.alpha_fn, ds)
state, metrics = train(cfg.epochs, rng, state)

# %% Polar plots
fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
seqs = [range(1, 1024), range(0, 1024, 6), twos_and_fives, primes[1025:2049]]
mi.plots.small_multiples(fnames[:3], seqs[:3], "polar_nats_and_sixes", 1, 3)
mi.plots.polar_plot(seqs[-1], "polar_primes")

# %% Hinton plots


# functions
def syrkis_plot(matrix, cfg, metric, x_scale="linear"):
    cols = matrix.shape[1] * 4
    X = matrix[: (matrix.shape[0] // cols) * cols]
    I = jnp.identity(cols).repeat(X.shape[0] // cols, axis=0)
    matrix = (I[:, :, None] * X[:, None, :]).mean(axis=0).T
    one, two = matrix.shape
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())  # type: ignore
    ax.yaxis.set_major_locator(plt.NullLocator())  # type: ignore
    for (y, x), w in np.ndenumerate(matrix):
        s = np.sqrt(w)
        # is last fg is blue
        # c = mi.utils.blue if y == 0 else fg
        rect = plt.Rectangle(  # type: ignore
            [x - s / 2, y - s / 2],  # type: ignore
            s,
            s,
            facecolor="black",
            edgecolor="black",  # type: ignore
        )
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    # ax.set_title(metric)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[0], step=cfg.epochs // 20))  # type: ignore
    plt.tight_layout()
    fname = "syrkis_" + mi.plots.fname_fn(cfg, metric)
    # plt.savefig(f"paper/figs/{fname}.svg")
    plt.show()


syrkis_plot(metrics["train_loss"], cfg, "Train Focal Loss")

# %% plots_test.py
#    test plots
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random


# %% functions
def test_hinton_plot():
    rng = random.PRNGKey(0)
    cfg = mi.kinds.Conf(base=2, n=2**14, l2=1e-4, dropout=0.1)
    x = random.uniform(rng, shape=(2, 2))
    y = random.uniform(rng, shape=(2, 2))
    params = mi.param.init_fn(rng, cfg, x, y)
    mi.plots.hinton_fig(params.blocks[0].head.query, cfg, "hinton_query")


if __name__ == "__main__":
    test_hinton_plot()

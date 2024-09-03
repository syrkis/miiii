# %% plots_test.py
#    test plots
# by: Noah Syrkis

# %% Imports
import miiiii as mi
from jax import random
import matplotlib.pyplot as plt
from oeis import A000040 as primes


# %% functions
def test_hinton_metric():
    rng = random.PRNGKey(0)
    cfg = mi.kinds.Conf(base=2, n=2**14, l2=1e-4, dropout=0.1)
    ds = mi.datum.data_fn(cfg.n, cfg.base, mi.numbs.base_ns)
    x = random.uniform(rng, shape=(20, 200))

    fig, ax = plt.subplots()
    mi.plots.hinton_metric({"train_loss": x}, "train_loss", ds)


def test_hinton_weight():
    rng = random.PRNGKey(0)
    x = random.uniform(rng, shape=(2, 20, 200))
    mi.plots.hinton_weight(x)


def test_curve_plot():
    rng = random.PRNGKey(0)
    x = random.uniform(rng, shape=(20, 200))
    mi.plots.curve_plot(x)


def test_spiral_plots():
    fnames = ["polar_nats", "polar_sixes", "polar_evens_and_fives", "polar_threes"]
    twos_and_fives = [range(0, 1024, 2), range(0, 1024, 5)]
    seqs = [range(1, 1024), range(0, 1024, 6), twos_and_fives, primes[1025:2049]]
    mi.plots.small_multiples(fnames[:3], seqs[:3], "polar_nats_and_sixes", 1, 3)
    mi.plots.polar_plot(seqs[-1], "polar_primes")

# main.py
#   miiiii main file
# by: Noah Syrkis

# %% Imports
import jax
from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from functools import partial
import miiiii
import sys,os
# set environment varialbe ENABLE_PJRT_COMPATIBILITY=1
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

# %% functions
if __name__ == "__main__":
    # config and init
    seed = 0
    cfg, (rng, key) = miiiii.get_conf(), random.split(random.PRNGKey(seed))
    train_data, valid_data = miiiii.prime_fn(cfg.n, cfg.base, miiiii.base_ns, rng)
    params = miiiii.init_fn(key, cfg, *train_data)

    # train
    apply_fn = miiiii.make_apply_fn(miiiii.vaswani_fn)
    train_fn, state = miiiii.init_train(apply_fn, params, cfg, miiiii.alpha_fn, train_data, valid_data)
    state, metrics = train_fn(cfg.epochs, rng, state)

    # evaluate
    # log_run(cfg, metrics, state.params)  # log run
    print(metrics[-1])  # print final metrics

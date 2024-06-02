# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from jax import random
import jax.numpy as jnp
from functools import partial
import argparse
import optax
from oeis import oeis
import src


# functions
def main():
    seq = oeis["A000040"]  # "A000040" is the sequence of prime numbers
    data_conf, model_conf = src.get_conf()
    rng, key = random.split(random.PRNGKey(0))

    number_system = partial(src.base_n, data_conf["base"])
    (train_x, train_y), _ = src.data_fn("primes", seq, data_conf["n"], number_system)

    params = src.init_fn(key, dict(**model_conf, len=train_x.shape[1]))

    apply_fn = src.make_apply_fn(src.vaswani_fn)
    loss_fn = src.make_loss_fn(apply_fn)
    grad_fn = src.make_grad_fn(loss_fn)

    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    update_fn = src.make_update_fn(opt)

    train_fn = src.make_train_fn(grad_fn, update_fn, train_x, train_y)
    state, losses = train_fn(params, opt_state, 5000)
    print(losses)


if __name__ == "__main__":
    main()

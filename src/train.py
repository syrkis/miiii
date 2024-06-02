# train.py
#   miiii train
# by: Noah Syrkis

# imports
import jax
from jax import random, grad, jit, value_and_grad
import optax
from tqdm import tqdm
import jax.numpy as jnp
from typing import List, Set, Tuple
from functools import partial
from oeis import oeis
from einops import rearrange


# functions
def make_loss_fn(apply_fn):
    @jit
    def loss_fn(params, x, y):  #  cross entropy loss
        logits = rearrange(apply_fn(params, x), "b t c -> (b t) c")
        labels = rearrange(y, "b t -> (b t)")
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return loss.mean()

    return loss_fn


def make_update_fn(opt):
    @jit
    def update_fn(params, grads, opt_state):
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_fn


def make_grad_fn(loss_fn):
    @jit
    def grad_fn(params, x, y):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        return loss, grads

    return grad_fn


def make_estimate_loss_fn(loss_fn):
    def estimate_loss(params, train_data, valid_data):
        train_loss = 0
        valid_loss = 0
        for i in range(10):
            train_loss += loss_fn(params, *next(train_data))
        for i in range(10):
            valid_loss += loss_fn(params, *next(valid_data))
        return train_loss / 10, valid_loss / 10

    return estimate_loss


# testing
if __name__ == "__main__":
    from param import init_fn
    from model import make_apply_fn, vaswani_fn
    from utils import get_conf
    from datum import data_fn
    from numbs import base_n

    # x, y = data_fn(oeis["A000040"], conf["n_samples"], partial(base_n, n=conf["base"]))
    rng, key = random.split(random.PRNGKey(0))
    train_data, valid_data, encode, decode, vocab = data_fn("ficciones", key, 16, 64)
    rng, key = random.split(rng)
    config = get_conf(
        in_d=len(vocab), out_d=len(vocab), len=next(train_data)[0].shape[1]
    )
    params = init_fn(key, config)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # train functions
    apply_fn = make_apply_fn(vaswani_fn)
    loss_fn = make_loss_fn(apply_fn)
    grad_fn = make_grad_fn(loss_fn)
    update_fn = make_update_fn(opt)

    # eval functions
    eval_apply_fn = jit(make_apply_fn(vaswani_fn))
    eval_loss_fn = make_loss_fn(eval_apply_fn)
    estimate_loss = make_estimate_loss_fn(eval_loss_fn)

    for i, (x, y) in zip(pbar := tqdm(range(10000)), train_data):
        loss, grads = grad_fn(params, x, y)
        params, opt_state = update_fn(params, grads, opt_state)
        if i % 500 == 0:
            train_loss, valid_loss = estimate_loss(params, train_data, valid_data)
            pbar.set_description(f"{train_loss:.3f} {valid_loss:.3f}")

    rng, key = random.split(rng)
    state = next(train_data)[0]
    for i in range(100):
        rng, key = random.split(rng)
        logits = eval_apply_fn(params, state[:, -32:])[:, -1, :]
        word = random.categorical(key, logits)[:, None]
        state = jnp.concatenate([state, word], axis=-1)

    print(decode(state[0].tolist()))

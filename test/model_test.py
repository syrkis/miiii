# %% model_test.py
#    test model
# by: Noah Syrkis

# %% Imports
import miiiii.model as model
import miiiii.kinds as kinds

from jax import numpy as jnp
from jax import random


# %% functions
def test_model():  # casdas
    # test with identity functions and null weights
    identity_ffwd = kinds.FFWD(
        w1=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        b1=jnp.array([0.0, 0.0]),
        w2=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        b2=jnp.array([0.0, 0.0]),
    )

    identity_head = kinds.Head(
        key=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        query=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        value=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        proj=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
    )

    identity_block = kinds.Block(head=identity_head, ffwd=identity_ffwd)

    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x = x.reshape((1, 2, 2))  # add embedding dimension

    x_hat = x + model.vaswani_head_fn(x, identity_head)
    x_hat = x_hat + model.ffwd_fn(x_hat, identity_ffwd)

    assert jnp.all(x_hat == model.vaswani_fn(x, identity_block))


def test_classify_fn():
    logits = jnp.array([-10.0, -5.0, 5.0, -5.0, -5.0, -5.0])
    pred = model.classify_fn(logits)
    assert jnp.all(pred == jnp.array([0, 0, 1, 0, 0, 0]).astype(jnp.int32))


def test_dropout_fn():
    x = jnp.array([1.0, 1.0, 1.0, 1.0])
    key = random.PRNGKey(0)
    x_hat, key = model.dropout_fn(key, x, 0.0)
    print(x_hat, x, sep="\n")
    assert jnp.all(x == x_hat)


def test_make_apply_fn():
    apply_fn = model.make_apply_fn(model.vaswani_fn)
    assert apply_fn is not None

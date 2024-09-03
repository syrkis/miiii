# %% Imports
import miiiii.datum as datum
import miiiii.numbs as numbs
import miiiii.kinds as kinds
import jax.numpy as jnp
import pytest

# %% #################################################################################################
# Test tests
# datum.primes_fn(10)
# datum.target_fn(10, 8, numbs.base_ns)


# %% #################################################################################################
# Test primes_fn


@pytest.mark.parametrize("input, expected", [(10, jnp.array([2, 3, 5, 7]))])
def test_primes(input, expected):  # confirm all primes less than n
    primes = datum.primes_fn(input)
    assert jnp.all(primes == expected), "wrong primes"  # test correct primes


# %% #################################################################################################
# Test source_fn

case_1_in = (10, 16, numbs.base_ns)
case_1_out = jnp.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
source_case_1 = (case_1_in, case_1_out)

case_2_in = (10, 8, numbs.base_ns)
case_2_out = jnp.array(
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 0], [1, 1]]
)
source_case_2 = (case_2_in, case_2_out)


@pytest.mark.parametrize("input, expected", [source_case_1, source_case_2])
def test_source_fn(input, expected):
    n, base, base_ns = input
    x = datum.source_fn(n, base, base_ns)
    assert jnp.all(x == expected), "wrong x"


# %% #################################################################################################
# Test data_fn

case_1_in = (10, 16, numbs.base_ns, None)
train_split = kinds.Datasplit(
    x=jnp.array([[0], [1], [2], [3], [4]]),
    y=jnp.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]]),
)
test_split = kinds.Datasplit(
    x=jnp.array([[5], [6], [7], [8], [9]]),
    y=jnp.array([[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]),
)
info = kinds.Datainfo(
    alpha=1 - jnp.array([3 / 5.0, 2 / 5.0, 2 / 5.0]), tasks=["2", "3", "prime"]
)
case_1_out = kinds.Dataset(train=train_split, valid=test_split, info=info)
data_case_1 = (case_1_in, case_1_out)


@pytest.mark.parametrize("input, expected", [data_case_1])
def test_data_fn(input, expected):
    n, base, base_ns, key = input
    ds = datum.data_fn(n, base, base_ns, key)
    assert jnp.allclose(ds.info.alpha, expected.info.alpha), "wrong alpha"
    assert jnp.all(ds.info.tasks == expected.info.tasks), "wrong tasks"
    assert jnp.all(ds.train.x == expected.train.x), "wrong train x"
    assert jnp.all(ds.train.y == expected.train.y), "wrong train y"
    assert jnp.all(ds.valid.x == expected.valid.x), "wrong valid x"
    assert jnp.all(ds.valid.y == expected.valid.y), "wrong valid y"

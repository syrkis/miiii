# %% algos_test.py
#    test algos
# by: Noah Syrkis

# %% Imports
import miiiii.algos as algos
import pytest


# %% Functions
@pytest.mark.parametrize(
    "n, expected",
    [
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (6, False),
        (7, True),
        (17, True),
        (18, False),
        (121212, False),
        (7919, True),
        (199933, True),
    ],
)
def test_trial_division(n, expected):
    assert algos.trial_division(n) == expected

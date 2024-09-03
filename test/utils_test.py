# %% utils_test.py
#    test utils
# by: Noah Syrkis

# %% Imports
import miiiii.utils as utils


# functions
def test_load_conf():
    conf = utils.load_conf()
    assert conf.base >= 1

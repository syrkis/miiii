# %% param_test.py
#    test param
# by: Noah Syrkis

# %% Imports
import miiiii.param as param
import miiiii.kinds as kinds
import miiiii.utils as utils
from jax import random


# functionos
def test_param():  # could be done much better
    rng = random.PRNGKey(0)
    cfg = kinds.Conf(base=2, n=2**14, l2=1e-4, dropout=0.1)
    x = random.uniform(rng, shape=(2, 2))
    y = random.uniform(rng, shape=(2, 2))
    params = param.init_fn(rng, cfg, x, y)

    assert len(params.blocks) == cfg.depth

    # save params
    utils.save_params(params, "test_params")
    params2 = utils.load_params("test_params")

    assert params.tok_emb.shape == params2.tok_emb.shape
    assert params.pos_emb.shape == params2.pos_emb.shape
    assert params.lm_head.shape == params2.lm_head.shape

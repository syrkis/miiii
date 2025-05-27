# tasks.py
#   miii data functions
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import Array, random
from oeis import oeis
from miiii.types import Conf, Dataset


def task_fn(key: Array, cfg: Conf, arg) -> Dataset:  # five me nanda or miiii task
    return miiii_fn(key, cfg, arg) if arg.task == "miiii" else nanda_fn(key, cfg, arg)


def repr_fn(x, p):  # represent x in base p (p is prime, and x is less than p**2)
    return jnp.stack((x // p, x % p, jnp.array(p).repeat(x.shape[0])), axis=-1).astype(jnp.int8)


### MULTI TRUE
def miiii_fn(key, cfg, arg):
    primes = jnp.array(oeis["A000040"][1 : cfg.p])  # potential prime factors
    factors = primes[primes < cfg.p]  # prime factors
    idxs = random.permutation(key, jnp.arange(cfg.p**2))  # permute the indices
    x = repr_fn(jnp.arange(cfg.p**2), cfg.p)  # x is the representation of the numbers
    y = (jnp.arange(cfg.p**2)[:, None] % factors[None, :]).astype(jnp.int8)
    y = y if arg.mods == "remainder" else (y == 0).astype(jnp.int8)  # this was a serious bug i just fixed it
    # y = y[random.permutation(random.split(key)[0], jnp.arange(cfg.p**2))] if cfg.shuffle else y
    primes = jnp.array(oeis["A000040"][1 : y.shape[1] + 1])
    mask = jnp.tile(jnp.arange(primes.max()), primes.size).reshape((primes.size, -1)) < primes[:, None]
    mask = mask if arg.mods == "remainder" else jnp.array(1)
    weight = 1 / jnp.log(mask.sum(-1))  # modify to ignore masked away tasks
    weight = weight.at[:4].set(0) if cfg.mask else weight  # mask first four tasks. maybe.
    return Dataset(x=x, y=y, idxs=idxs, mask=mask, weight=weight, frac=cfg.train_frac, primes=jnp.array(factors))


# nanda task  ################################################################
def nanda_fn(key, cfg: Conf, arg) -> Dataset:
    # modular adition modulo prime
    a = jnp.arange(cfg.p).repeat(cfg.p)
    b = jnp.tile(jnp.arange(cfg.p), cfg.p)
    y = (a + b) % cfg.p
    data = jnp.stack([a, b, jnp.array(cfg.p).repeat(cfg.p**2), y], axis=-1)
    idxs = random.permutation(key, len(data))
    x = data[:, :-1]
    y = data[:, -1]
    y = (y == 0).astype(jnp.int8) if arg.mods == "divisible" else y
    return Dataset(x=x, y=y, idxs=idxs, frac=cfg.train_frac, mask=jnp.array(1), primes=jnp.array([cfg.p]))

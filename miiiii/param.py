# init.py
#   miiii init
# by: Noah Syrkis

# imports
import miiiii as mi
from jax import random, Array
import jax.numpy as jnp
from oeis import A000040


# init functions
def init_head_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Head:
    h, d = cfg.heads, cfg.latent_dim
    keys = random.split(rng, 4)

    key = random.uniform(keys[0], shape=(h, d, d // h), minval=-cfg.theta, maxval=cfg.theta)
    query = random.uniform(keys[1], shape=(h, d, d // h), minval=-cfg.theta, maxval=cfg.theta)
    value = random.uniform(keys[2], shape=(h, d, d // h), minval=-cfg.theta, maxval=cfg.theta)
    # proj = random.uniform(keys[3], shape=(h * d // h, d), minval=-cfg.theta, maxval=cfg.theta)

    return mi.kinds.Head(query=query, key=key, value=value)  # proj=proj)


def init_ffwd_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.FFWD:
    keys = random.split(rng)
    w1 = random.uniform(keys[0], shape=(cfg.latent_dim, cfg.latent_dim), minval=-cfg.theta, maxval=cfg.theta)
    w2 = random.uniform(keys[1], shape=(cfg.latent_dim, cfg.latent_dim), minval=-cfg.theta, maxval=cfg.theta)
    b1 = jnp.zeros(cfg.latent_dim)
    b2 = jnp.zeros(cfg.latent_dim)
    return mi.kinds.FFWD(w1=w1, b1=b1, w2=w2, b2=b2)


def init_block_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Block:
    keys = random.split(rng)
    head = init_head_fn(keys[0], cfg)
    ffwd = init_ffwd_fn(keys[1], cfg)
    params = mi.kinds.Block(head=head, ffwd=ffwd)
    return params


def init_fn(rng: Array, cfg: mi.kinds.Conf) -> mi.kinds.Params:  # x: Array, y: Array) -> mi.kinds.Params:
    keys = random.split(rng, 3 + cfg.depth)
    params = mi.kinds.Params(
        tok_emb=random.uniform(keys[0], shape=(cfg.vocab_size, cfg.latent_dim), minval=-cfg.theta, maxval=cfg.theta),
        pos_emb=random.uniform(keys[1], shape=(cfg.seq_len, cfg.latent_dim), minval=-cfg.theta, maxval=cfg.theta),
        lm_head=random.uniform(keys[2], shape=(cfg.latent_dim, y_fn(cfg)), minval=-cfg.theta, maxval=cfg.theta),
        blocks=[init_block_fn(key, cfg) for key in keys[3:]],
    )
    return params


def y_fn(cfg: mi.kinds.Conf) -> int:
    primes = jnp.array(A000040[1 : cfg.n * 2])
    primes = primes[primes < jnp.sqrt(cfg.n)]
    return primes.shape[0] + 1 if cfg.task == "prime" else cfg.vocab_size

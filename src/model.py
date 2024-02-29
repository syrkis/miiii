# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import numpy as np

from functools import partial
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple, Union


# forward functions
def apply_fn(params, x):
    x = embed_fn(params, x)
    for block in params['blocks']:
        x = head_fn(block['head'], x) + mlp_fn(block['mlp'], x)
    logits = x @ params['lm_head']  # B x T x V
    return logits[:, -1, :]

def head_fn(params, x):
    head_apply   = partial(head_apply_fn, x=x)
    all_heads_fn = vmap(head_apply, in_axes=(0,0,0,0))
    x            = all_heads_fn(*params)
    return jnp.mean(x, axis=0)

def head_apply_fn(key_param, val_param, alpha, beta, x):
    k  = x @ key_param                            # B x T x H
    q  = x @ val_param                            # B x T x H
    z  = q @ rearrange(k, "b t h -> b h t")       # k.transpose(0, 2, 1)          # B x T x T
    z /= jnp.sqrt(key_param.shape[1])             # divide by sqrt to normalize
    z  = (alpha * np.eye(x.shape[1]))[None, :, :] #  + beta * x                   # - gamma * C  # <-- shaped attention
    return z @ x
    
def mlp_fn(params, x):  
    x = x @ params['dense1'] + params['bias1']
    x = jax.nn.relu(x)  # TODO: maybe switch activation
    x = x @ params['dense2'] + params['bias2']
    return x

def embed_fn(params, x):
    z  = params['tok_emb'][x]                       # tok embeddings
    z += params['pos_emb'][jnp.arange(x.shape[1])]  # pos embeddings
    return z

# init functions
def init_head_fn(rng, conf):
    rngs      = jax.random.split(rng, conf['n_heads'])
    keys      = jax.random.normal(rngs[0], shape=(conf['n_heads'], conf['emb_dim'], conf['emb_dim'] // conf['n_heads'])) * conf['scale']
    values    = jnp.zeros_like(keys)
    return keys, values, jnp.ones((conf['n_heads'],)), jnp.zeros((conf['n_heads'],))  # ALPHA AND BETA

def init_mlp_fn(rng, conf):
    rng, key1, key2 = jax.random.split(rng, 3)
    params = {
        'dense1' : jax.random.normal(key1, shape=(conf['emb_dim'], conf['emb_dim'])) * conf['scale'],
        'bias1'  : jnp.zeros((conf['emb_dim'],)),
        'dense2' : jax.random.normal(key2, shape=(conf['emb_dim'], conf['emb_dim'])) * conf['scale'],
        'bias2'  : jnp.zeros((conf['emb_dim'],)),
        }
    return params

def init_block_fn(rng, conf):
    rng, key1, key2 = jax.random.split(rng, 3)
    params = {
        'head' : init_head_fn(key1, conf),
        'mlp'  : init_mlp_fn(key2, conf),
        }
    return params

def init_fn(rng, conf):  # unmasking (like show that is is a prime, and ask it to finish it) online primes in data.
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    params = {
        'tok_emb' : jax.random.normal(key1, shape=(conf['base'], conf['emb_dim'])) * conf['scale'],
        'pos_emb' : jax.random.normal(key2, shape=(conf['block_size'], conf['emb_dim'])) * conf['scale'],
        'lm_head' : jax.random.normal(key3, shape=(conf['emb_dim'], 1)) * conf['scale'],  # one for classification.
        'blocks'  : [init_block_fn(key1, conf) for _ in range(conf['n_layers'])],
        }
    return params
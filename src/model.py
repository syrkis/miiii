# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

from functools import partial
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple, Union


# forward functions
def apply_fn(params, x):
    x = embed_fn(params, x)
    for block in params['blocks']:
        x = head_fn(block['head'], x) + mlp_fn(block['mlp'], x)
    logits = x @ params['lm_head']                       # B x T x V
    return logits[:, -1, :]

def head_fn(params, x):
    x = [head_apply_fn(params[f'head_{i}'], x) for i in range(len(params))]
    x = jnp.stack(x, axis=-1).mean(axis=-1)
    return x

def head_apply_fn(params, x):
    k  = x @ params['key']                      # B x T x H
    q  = x @ params['query']                    # B x T x H
    z  = q @ rearrange(k, "b t h -> b h t")  # k.transpose(0, 2, 1)               # B x T x T
    z /= jnp.sqrt(params['key'].shape[1])       # divide by sqrt to normalize
    z  = params['alpha'] * np.eye(x.shape[1]) # +  params['beta'] * x              # - gamma * C  # <-- shaped attention
    return z @ x
    
def mlp_fn(params, x):  
    x = x @ params['dense1'] + params['bias1']
    x = jax.nn.relu(x)  # TODO: maybe switch activation
    x = x @ params['dense2'] + params['bias2']
    return x

def embed_fn(params, x):
    n  = x.shape[1]                              # num toks in sample
    x  = params['tok_emb'][x]              # tok embeddings
    x += params['pos_emb'][jnp.arange(n)]  # pos embeddings
    return x


# init functions
def init_head_fn(rng, conf):  # emb_dim, n_heads, scale):
    head_size = conf['emb_dim'] // conf['n_heads']
    rng, key  = jax.random.split(rng)
    params = {} 
    for i in range(conf['n_heads'] ):
        params[f'head_{i}'] = {
            'key'   : jax.random.normal(key, shape=(conf['emb_dim'] , head_size)) * conf['scale'] ,
            'query' : jnp.zeros((conf['emb_dim'] , head_size)),
            'alpha' : jnp.array(1),
            'beta'  : jnp.array(0),
            }
    return params

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

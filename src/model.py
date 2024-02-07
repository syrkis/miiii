# model.py
#   miii model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

from functools import partial
from tqdm import tqdm
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple, Union

from src.data import decode

# functions
def head_fn(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    x = [head_apply_fn(params['head'][f'head_{i}'], x) for i in range(len(params['head']))]
    x = jnp.concatenate(x, axis=-1)
    x = jnp.dot(x, params['proj'])
    return x

def head_apply_fn(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    # x: B x T x C
    B, T, C = x.shape
    # tril = jnp.nan_to_num(jnp.absolute(jnp.tril(jnp.ones((T, T))) - 1) * (-jnp.inf), nan=0)
    # mask = jnp.nan_to_num(jnp.triu(jnp.ones((T, T))) * (-jnp.inf), nan=0)
    H = params['key'].shape[1]
    k = jnp.dot(x, params['key'])       # B x T x H
    q = jnp.dot(x, params['query'])     # B x T x H
    wei = q @ k.transpose(0, 2, 1)      # B x T x T
    wei /= jnp.sqrt(H)                  # normalise
    # wei += tril                         # mask future
    wei = jax.nn.softmax(wei, axis=-1)  # B x T x T
    v = jnp.dot(x, params['value'])     # B x T x H
    out = wei @ v                       # B x T x H
    return out

def init_head_fn(rng, embed_dim, n_heads, scale):
    head_size = embed_dim // n_heads
    rng, key_key, key_value, key_query = jax.random.split(rng, 4)
    params = {} 
    for i in range(n_heads):
        params[f'head_{i}'] = {
            'key':   jax.random.normal(key_key,   shape=(embed_dim, head_size)) * scale,
            'value': jax.random.normal(key_value, shape=(embed_dim, head_size)) * scale,
            'query': jax.random.normal(key_query, shape=(embed_dim, head_size)) * scale,
            }
    return params

def ffwd_fn(params, x):
    out = jax.nn.relu(x @ params['dense1'] + params['bias1'])
    out = out @ params['dense2'] + params['bias2']
    return out

def init_ffwd_fn(rng, embed_dim, scale=1e-2):
    rng, key1, key2 = jax.random.split(rng, 3)
    params = {
        'dense1': jax.random.normal(key1, shape=(embed_dim, 4 * embed_dim)) * scale,
        'bias1': jax.random.normal(key1, shape=(4 * embed_dim,)) * scale,
        'dense2': jax.random.normal(key2, shape=(4 * embed_dim, embed_dim)) * scale,
        'bias2': jax.random.normal(key2, shape=(embed_dim,)) * scale,
        }
    return params


def layer_norm_fn(params, x, eps=1e-6):
    gamma, beta = params['gamma'], params['beta']
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    out = (x - mean) / (std + eps)
    out = out * gamma + beta
    return out

def init_layer_norm_fn(embed_dim):
    params = {
        'gamma': jnp.ones((embed_dim,)),
        'beta': jnp.zeros((embed_dim,)),
        }
    return params


def init_block_fn(rng, embed_dim, n_heads, scale):
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    params = {
        'head': init_head_fn(key1, embed_dim, n_heads, scale),
        'ffwd': init_ffwd_fn(key2, embed_dim, scale),
        'proj': jax.random.normal(key3, shape=(embed_dim, embed_dim)) * scale,
        'ln1': init_layer_norm_fn(embed_dim),
        'ln2': init_layer_norm_fn(embed_dim),
        }
    return params

def block_fn(params, x):
    x = layer_norm_fn(params['ln1'], x)
    x += head_fn(params, x)
    x = layer_norm_fn(params['ln2'], x)
    x += ffwd_fn(params['ffwd'], x)
    return x

@jit
def apply_fn(params, xb):
    B, T = xb.shape
    tok_embs = params['tok_embedding'][xb]              # B x T x C
    pos_embs = params['pos_embedding'][jnp.arange(T)]   # T x C
    x = tok_embs + pos_embs
    for block in params['blocks']:
        x = block_fn(block, x)
    x = layer_norm_fn(params['layer_norm'], x)
    logits = x @ params['lm_head']                       # B x T x V
    return logits


def init_fn(rng, config):
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    params = {
        'tok_embedding': jax.random.normal(key1, shape=(config['vocab_size'], config['embed_dim'])) * config['scale'],
        'pos_embedding': jax.random.normal(key2, shape=(config['block_size'], config['embed_dim'])) * config['scale'],
        'lm_head': jax.random.normal(key3, shape=(config['embed_dim'], config['vocab_size'])) * config['scale'],
        'blocks': [init_block_fn(key1, config['embed_dim'], config['n_heads'], scale=config['scale']) for _ in range(config['n_layers'])],
        'layer_norm': init_layer_norm_fn(config['embed_dim']),
        }
    return params


def loss_fn(params, xb, yb):
    # we cant to minimise cross entropy
    logits = apply_fn(params, xb) # B x T x C
    B, T, C = logits.shape
    yb = yb.reshape(-1)
    logits = logits.reshape(B * T, C)
    logits = jnp.clip(logits, -100, 100)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(yb, C)))
    return loss


def generate_fn(rng, params, idx, block_size, length=100, temperature=0.6, verbose=False):
    pbar = tqdm(range(length)) if not verbose else range(length)
    # pad idx with 0s to block_size to the left
    # idx = jnp.pad(idx, (block_size - idx.shape[1], 0), mode='constant', constant_values=0)
    for _ in pbar:
        rng, key = jax.random.split(rng)
        logits = apply_fn(params, idx[:, -block_size:])         # B x T x C
        logits = logits[:, -1, :] / temperature                 # B x C
        idx_new = jax.random.categorical(key, logits)[:, None]  # B x 1
        idx = jnp.concatenate([idx, idx_new], axis=1)           # B x T + 1
        if verbose:
            print(decode(idx[0].tolist()))
    return idx

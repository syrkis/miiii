# %% main.py
#   miiii notebook
# by: Noah Syrkis

# Imports
from jax import random, value_and_grad, vmap, nn, tree
import jax.numpy as jnp
import miiii as mi
import mlxp
from jaxtyping import Array
from functools import partial
import optax
from miiii.types import Params


# ds = datasets.load_iris()


def init_fn(rng, cfg, ds) -> Params:
    a, k = nn.initializers.glorot_normal(), random.split(rng, 5)
    s: tuple = ((cfg.p + 1, cfg.d), (3, cfg.d), (cfg.d, cfg.d * 4), (cfg.d * 4, cfg.d), (ds.t, cfg.d, cfg.p))
    return Params(tok=a(k[0], s[0]), pos=a(k[1], s[1]), w_i=a(k[2], s[2]), w_o=a(k[3], s[3]), out=a(k[4], s[4]))


def apply(params, x) -> Array:  # TODO: perhaps attention circuit is needed here.
    z: Array = params.tok.take(x, axis=0) + params.pos.take(jnp.arange(3), axis=0)
    z = z + jnp.dot(nn.relu(jnp.dot(z, params.w_i)), params.w_o)
    return jnp.dot(z.sum(1), params.out)


@value_and_grad
def grad_fn(params: Array, x: Array, y: Array, mask: Array) -> Array:
    logits: Array = apply(params, x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask)
    return losses.mean()


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    opt = optax.adam(ctx.config.lr)
    rng = random.PRNGKey(ctx.config.seed)
    ds = mi.tasks.task_fn(rng, ctx.config.p)
    params = init_fn(rng, ctx.config, ds)
    opt_state: Params = opt.init(params)  # type: ignore
    for i in range(ctx.config.epochs):
        loss, grad = grad_fn(params, ds.x, ds.y, ds.mask)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if i % ctx.config.tick == 0:
            print(loss)


if __name__ == "__main__":
    main()


# print(loss_fn(w, x, y))

exit()

"""
@partial(vmap, in_axes=(None, None, 0))
def apply(key: Array, params: mi.types.Params, x) -> Array:
    x = jnp.take(params.tok_emb, x, axis=0) + jnp.take(params.pos_emb, jnp.arange(x.shape[0]), axis=0)
    x = jnp.dot(nn.relu(jnp.dot(x, params.w_i)), params.w_o)  # z: seq_len x emb_dim
    return jnp.dot(x[-1], params.out_emb)  # TODO: mask some things to zero?


def loss_fn(params: mi.types.Params, x, y, mask, rng: Array) -> Array:
    logits: Array = apply(rng, params, x)
    losses: Array = cross_entropy(logits, y, mask)  # / ds.task  # normalize by n-ary task
    return losses.mean()  # mean across tasks (weigted by expected n-ary classification loss)


# %% HELPERS
@partial(vmap, in_axes=(1, 1, 0))
def cross_entropy(logits: Array, y: Array, mask: Array) -> Array:  # mean cross entropy acroess samples
    return optax.softmax_cross_entropy_with_integer_labels(logits, y, where=mask).mean()


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    rngs = random.split(random.PRNGKey(ctx.config.seed), 4)
    ds = mi.tasks.task_fn(rngs[0], ctx.config)  # correct up to here
    opt = optax.adam(ctx.config.lr)
    params = mi.model.init_fn(rngs[0], ctx.config, ds)
    opt_state = opt.init(params)  # type: ignore
    grad_fn = value_and_grad(loss_fn)

    for i in range(ctx.config["epochs"]):
        loss, grad = grad_fn(params, ds.x, ds.y, ds.mask, rngs[0])
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(loss)
    # print(ds.x.T[:2], ds.y.T, sep="\n\n")
    # state, (loss, scope) = mi.train.train_fn(rngs[3], ctx.config, ds, state, opt)
    # logits, z = mi.model.apply(rngs[0], state.params, ds.x)
    # print(logits.argmax(-1).deeper)
    # print(ds.y.deeper)
    # print(ds.y[: ctx.config["p"] * 2].T.deeper)
    # print(scope.cce.T.deeper)
    # print(scope.acc.T.deeper)
    # print(ds.task.deeper)
    # log_fn(ctx, ds, state, loss)


def log_fn(ctx, ds: mi.types.Dataset, state: mi.types.State, loss):
    fns = [mi.plots.plot_params, mi.plots.plot_x, mi.plots.plot_y, mi.plots.plot_log]
    [fn(cfg=ctx.config, ds=ds, params=state.params) for fn in fns]

    for idx, tick in enumerate(loss):
        for jdx, entry in enumerate(tick):
            ctx.logger.log_metrics(dict(epoch=idx * loss.shape[1] + jdx, loss=entry.item()), "train")


if __name__ == "__main__":
    main()

"""

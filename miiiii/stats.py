# stats.py
#   stats functions for miiiii
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from aim import Run
from jax import Array


# functions
def horizontal_mean_pooling(x: Array, width: int = 3) -> Array:
    """Rolling mean array. Shrink to be rows x rows * width."""
    x = x[:, : (x.shape[1] // (x.shape[0] * width)) * (x.shape[0] * width)]
    i = jnp.eye(x.shape[0] * width).repeat(x.shape[1] // (x.shape[0] * width), axis=-1)
    z = (x[:, None, :] * i[None, :, :]).sum(axis=-1)
    return z / (x.shape[1] // (x.shape[0] * width))


def track_metrics(metrics, ds, cfg):
    run = Run(experiment="miiiii")
    run["cfg"] = cfg.__dict__

    for epoch in range(cfg.epochs):
        for idx, task in enumerate(ds.info.tasks):
            for split in ["train", "valid"]:
                to_log = {k: v[epoch][idx] for k, v in metrics[split].items()}
                run.track(
                    to_log, epoch=epoch + 1, context={"task": task, "split": split}
                )

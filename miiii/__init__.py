from . import tasks  # noqa
from . import utils  # noqa
from . import train  # noqa
from . import types  # noqa
from . import model  # noqa
from .plots import plot_y, plot_x  # noqa
# import lovely_jax as lj

# lj.monkey_patch()
plots = [plot_y, plot_x]


__all__ = [
    "types",
    "model",
    "tasks",
    "utils",
    "plots",
    "train",
    "plots",
]

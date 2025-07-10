from . import tasks  # noqa
from . import utils  # noqa
from . import train  # noqa
from . import types  # noqa
from . import model  # noqa
from . import optim  # noqa

from .plots import plot_y, plot_x  # noqa

plots = [plot_y, plot_x]


__all__ = [
    "types",
    "model",
    "tasks",
    "utils",
    "plots",
    "train",
    "optim",
    "plots",
]

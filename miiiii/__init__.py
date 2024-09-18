import os

os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["MPS_WATCHDOG_TIMEOUT"] = "60000"
os.environ["JAX_DEBUG_NANS"] = "True"

from . import kinds
from . import prose
from . import prime
from . import utils
from . import plots
from . import train
from . import model


__all__ = [
    "kinds",
    "prose",
    "prime",
    "utils",
    "plots",
    "train",
    "model",
]

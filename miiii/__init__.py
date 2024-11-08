import os

os.environ["JAX_ENABLE_X64"] = "True"  # Add this line
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["MPS_WATCHDOG_TIMEOUT"] = "60000"
os.environ["JAX_DEBUG_NANS"] = "True"


from . import scope  # noqa
from . import tasks  # noqa
from . import utils  # noqa
from . import plots  # noqa
from . import train  # noqa
from . import model  # noqa


__all__ = [
    "model",
    "scope",
    "tasks",
    "utils",
    "plots",
    "train",
]

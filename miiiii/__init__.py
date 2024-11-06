import os
from ctypes.macholib import dyld  # type: ignore

dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")
os.environ["JAX_ENABLE_X64"] = "True"  # Add this line
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["MPS_WATCHDOG_TIMEOUT"] = "60000"
os.environ["JAX_DEBUG_NANS"] = "True"


from . import scope
from . import tasks
from . import utils
from . import plots
from . import train
from . import model


__all__ = [
    "model",
    "scope",
    "tasks",
    "utils",
    "plots",
    "train",
]

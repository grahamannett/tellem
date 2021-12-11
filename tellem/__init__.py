import os

__version__ = "0.0.1"
_tellem_backend = os.getenv("TELLEM", "torch")


from tellem.engine.torch import *
from tellem.engine.uses_backend import uses_backend


if _tellem_backend != "torch":
    raise NotImplementedError()

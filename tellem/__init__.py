import os

__version__ = "0.0.1"
_tellem_backend = os.getenv("TELLEM", "torch")


import tellem.engine.torch as backend

Capture = backend.Capture

if _tellem_backend != "torch":
    raise NotImplementedError()

from tellem.types import Model


class ImplementationBase:
    def __init__(self, model: Model = None, **kwargs):
        self.model = model

    def _check_backend(self):
        pass
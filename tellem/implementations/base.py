from tellem.types import Model
from functools import reduce, wraps


class ImplementationBase:
    repr_info = {}

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"{self.repr_info}"

    def __tellem_function__(self, *args, **kwargs):
        """method that allows for you to explicitly show to use class in most simple way possible and it will run tests against it

        Raises:
            NotImplementedError: [description]
        """

        raise NotImplementedError("method not completed for function")

    def check_backend(self):
        pass

    def _funcs_reduce(self, funcs, **kwargs):
        return [func(**kwargs) for func in funcs]


class Usage:
    def __init__(self, order):
        self.order = []

    def after(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

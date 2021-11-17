from functools import singledispatch, wraps


def uses_backend(backend):
    backends_ = {"torch": uses_torch}
    return backends_[backend.__name__]


def uses_torch(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # setattr(self, func.__name__, func)
        return func(self, *args, **kwargs)

    return wrapper

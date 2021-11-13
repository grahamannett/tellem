from typing import Any, Callable

from tellem.implementations import ImplementationBase
from tellem.types import Model, Tensor


class FastGradientSignMethod(ImplementationBase):
    def __init__(self, model: Model = None, loss_func: Callable[..., Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    def generate_attack(self, x: Tensor, y: Tensor, **kwargs):
        pass

    def __tellem_function__(self, *args, **kwargs):
        return super().__tellem_function__(*args, **kwargs)

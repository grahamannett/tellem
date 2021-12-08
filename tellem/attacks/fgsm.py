from typing import Any, Callable

import torch

from tellem.implementations import ImplementationBase
from tellem.types import Model, Tensor


class FastGradientSignMethod(ImplementationBase):
    def __init__(self, model: Model = None, loss_func: Callable[..., Any] = None, eps: float = 0.01, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.loss_func = loss_func
        self.eps = eps

    def __call__(self, x: Tensor, y: Tensor = None, *args: Any, **kwds: Any) -> Any:
        return self.generate_attack(x, y)

    def generate_attack(self, x: Tensor, y: Tensor = None, **kwargs):
        y_pred = self.model(x)

        y = y_pred if y is None else y

        # breakpoint()
        loss = self.loss_func(y_pred, y)
        self.model.zero_grad()
        loss.backward()

        x_grad = x.grad.data
        sign_x_grad = x_grad.sign()

        perturbed_x = x + self.eps * sign_x_grad
        return perturbed_x

    def test_attack(self, x, y):
        pass

    def __tellem_function__(self, *args, **kwargs):
        return super().__tellem_function__(*args, **kwargs)

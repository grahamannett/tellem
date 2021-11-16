from functools import singledispatchmethod
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import nn

from tellem.types import RemovableHandle, Tensor


class CaptureManager:
    def __init__(self) -> None:
        self._capture = {}

    def __getitem__(self, key):
        return self._capture[key]

    def __setitem__(self, key, value):
        self._capture[key] = value

    def attach(self, *args, **kwargs):
        pass

    def detach(self):
        for key in self._capture.keys():
            self._capture[key] = None


class Capture:
    """capture a layer/module of a model.  either activations or gradients"""

    def __init__(self, model: nn.Module, layer: str):
        self.model = model
        self.layer = layer

        self.module = None
        self._activations = False
        self._gradients = False

        self.activations = None
        self.gradients = None

        self.hooks = []
        self.removed_hooks = []

    @singledispatchmethod
    def get_layer(self, layer):
        raise NotImplementedError("Cannot get layer")

    @get_layer.register
    def _(self, layer: str):
        return getattr(self.model, layer)

    @get_layer.register
    def _(self, layer: int):
        return list(self.model.named_children())[layer][1]

    def _default_activations_hook(self, module, inputs, outputs):
        self.activations = outputs

    def _default_gradients_hook(self, grad):
        self.grad = grad

    def capture_activations(self, hook: Callable[..., Any] = None):
        if hook is None:
            hook = self._default_activations_hook

        self._activations = True
        module = dict(self.model.named_modules())[self.layer]
        self.hooks.append(module.register_forward_hook(hook))
        return self

    def capture_gradients(self, hook: Callable[..., Any] = None):
        if hook is None:
            hook = self._default_gradients_hook

        self._gradients = True
        hook_ = self.activations.register_hook(hook)
        self.hooks.append(hook_)
        return self

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove()


class CaptureHandler:
    """base class for capturing intermediate values from pytorch model"""

    def __init__(self, module: nn.Module, keep_input: bool = False, keep_output: bool = False):
        self.keep_input = keep_input
        self.keep_output = keep_output

        self.handle = None

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if self.keep_input:
            self.inputs = inputs
        if self.keep_output:
            self.output = output

    def __del__(self):
        self.handle.remove()


class CaptureActivationsHandler(CaptureHandler):
    def __init__(self, module: nn.Module, keep_input: bool = False, keep_output: bool = True):
        super().__init__(module, keep_input=keep_input, keep_output=keep_output)
        self.handle = module.register_forward_hook(self)


class CaptureGradientsHandler(CaptureHandler):
    def __init__(self, module: nn.Module, keep_input: bool = True, keep_output: bool = True):
        super().__init__(module, keep_output=keep_output)
        self.handle = module.register_backward_hook(self)

from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch import nn

from tellem.types import Model, RemovableHandle, Tensor


class Capture:
    """capture a layer/module of a model.  either activations or gradients"""

    def __init__(self, model: nn.Module, layer: str):
        self.model = model
        self.layer = layer

        self.module = None
        self._capture_activations = False
        self._capture_gradients = False

        self.activations = None
        self.gradients = None

        # for holding the old values
        self._activations = []
        self._gradients = []

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

    def capture_activations(self, hook: Callable[..., Any] = None) -> Capture:
        if hook is None:

            def hook(module, inputs, outputs):
                if self.activations is not None:
                    self._activations.extend(self.activations)

                self.activations = outputs.detach()

        self._capture_activations = True
        module = dict(self.model.named_modules())[self.layer]
        self.hooks.append(module.register_forward_hook(hook))
        return self

    def capture_gradients(self, hook: Callable[..., Any] = None) -> Capture:
        """use this function to get gradients AFTER you have done something like model(x)

        Args:
            hook (Callable[..., Any], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if hook is None:

            def hook(grad):
                self.gradients = grad

        self._capture_gradients = True
        self.activations.requires_grad_(True)
        hook_ = self.activations.register_hook(hook)
        self.hooks.append(hook_)
        return self

    def remove(self):
        while self.hooks:
            hook = self.hooks.pop()
            hook.remove()
            self.removed_hooks.append(hook)
        self._activations = []
        self._gradients = []
        self.activations = None
        self.gradients = None

    def gather(self) -> torch.Tensor:
        self._activations.extend(self.activations)
        return torch.stack(self._activations)

    def __del__(self):
        self.remove()


class CaptureManager:
    def __init__(self, model: Model, start_attached: bool = False) -> None:
        self.model = model
        self.captures = {}

        self._start_attached = start_attached

    def __getitem__(self, key):
        return self.captures[key]

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.detach()

    @singledispatchmethod
    def __setitem__(self, key, value):
        raise NotImplementedError("Cant set this for CaptureManager")

    @__setitem__.register
    def _(self, key: str, value: Capture):
        self.captures[key] = value

    # not sure if i want this function
    def __call__(self, x: Tensor, **kwargs):
        preds = self.model(x)

        activations = {}

        for key, capture in self.captures.items():
            activations[key] = getattr(capture, "activations")

        return activations, preds

    def capture_layer(self, layer: str, attach: bool = True) -> CaptureManager:
        capture = Capture(self.model, layer)

        if attach:
            capture.capture_activations()
        self.__setitem__(layer, capture)
        return self

    def capture(self, layer_types: List[torch.nn.Module] = None) -> CaptureManager:
        for name, layer in self.model.named_modules():

            if (layer_types is not None) and (isinstance(layer, layer_types) == False):
                continue

            self.capture_layer(name, attach=self._start_attached)

        return self

    def capture_layers_of_type(self, layers: Sequence[nn.Module]) -> CaptureManager:
        return self.capture(layer_types=layers)

    def attach(self, *args, **kwargs) -> CaptureManager:
        for key in self.captures.keys():
            self.capture_layer(key)

        return self

    def detach(self) -> CaptureManager:
        for key, capture in self.captures.items():
            if capture:
                capture.remove()
            self.captures[key] = None

        return self

    def update(self, epoch: int):
        pass

    def batch_update(self, **kwargs):
        pass


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

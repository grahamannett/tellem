from __future__ import annotations

from typing import Dict
from functools import singledispatchmethod
from typing import Any, Callable, List, Sequence
from tellem.engine.torch.utils import flatten_module

import torch
from torch import nn


class Capture:
    """capture a layer/module of a model

    _activations = List[torch.Tensor] old values
    _gradients = List[torch.Tensor] old values
    self.activations = current activations
    self.gradients = current grads
    """

    def __init__(
        self,
        model: nn.Module,
        layer: str,
        keep_activations: bool = True,
        keep_gradients: bool = True,
        **kwargs,
    ):
        self.model = model
        self.layer = layer

        self.module = None
        self._capture_activations = False
        self._capture_gradients = False

        self._keep_activations = keep_activations
        self._keep_gradients = keep_gradients

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._reset_vals()

    def __repr__(self) -> str:
        return f"capture:{self.layer}, hooks:{self.hooks}"

    @singledispatchmethod
    def get_layer(self, layer):
        raise NotImplementedError("Cannot get layer")

    @get_layer.register
    def _(self, layer: str):
        return getattr(self.model, layer)

    @get_layer.register
    def _(self, layer: int):
        return list(self.model.named_children())[layer][1]

    @property
    def active(self):
        return len(self.hooks) > 0

    @active.setter
    def active(self, flag):
        _change_status = {True: self.attach, False: self.remove}[flag]
        _change_status()

    def capture_activations(self, hook: Callable[..., Any] = None) -> Capture:
        if hook is None:

            def hook(module, inputs, outputs):
                if (self.activations is not None) and self._keep_activations:
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
                if self.gradients is not None and self._keep_gradients:
                    self._gradients.extend(self.gradients)

                self.gradients = grad

        self._capture_gradients = True
        self.activations.requires_grad_(True)
        hook_ = self.activations.register_hook(hook)
        self.hooks.append(hook_)
        return self

    def attach(self):
        self.capture_activations()

    def remove(self):
        while self.hooks:
            hook = self.hooks.pop()
            hook.remove()

        self._reset_vals()

    def gather(self) -> torch.Tensor:
        self._activations.extend(self.activations)
        return torch.stack(self._activations)

    def _reset_vals(self):
        self._activations = []
        self._gradients = []
        self.activations = None
        self.gradients = None

    def __del__(self):
        self.remove()


class CaptureManager:
    def __init__(
        self, model: nn.Module, return_preds: bool = True, start_attached: bool = False, **kwargs
    ) -> None:
        self.model = model

        self.captures = {}
        self._start_attached = start_attached
        self.return_preds = return_preds

    @classmethod
    def go(cls, model: nn.Module, keep_activations=False, return_preds: bool = True, **kwargs):
        return cls(
            model,
            return_preds=return_preds,
        ).capture(keep_activations=keep_activations)

    def __getitem__(self, key):
        return self.captures[key]

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.detach()

    def __del__(self):
        self.detach()

    @singledispatchmethod
    def __setitem__(self, key, value):
        raise NotImplementedError("Cant set this for CaptureManager")

    @__setitem__.register
    def _(self, key: str, value: Capture):
        self.captures[key] = value

    def __call__(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        preds = self.model(x)

        activations = {key: capture.activations for key, capture in self.captures.items()}

        if self.return_preds:
            activations["predictions"] = preds

        return activations

    def __len__(self) -> int:
        """return the number of captures"""
        return len(self.captures)

    def list(self) -> Sequence[Any]:
        return list(self.captures.keys())

    def acts(self, x: torch.Tensor, return_preds: bool = False):
        return self.__call__(x, return_preds=return_preds)

    def capture_layer(self, layer: str, start_attached: bool = True, **kwargs) -> CaptureManager:
        capture = Capture(self.model, layer, **kwargs)

        if start_attached:
            capture.capture_activations()
        self.__setitem__(layer, capture)
        return self

    def capture_layers_of_type(self, layers: Sequence[nn.Module], **kwargs) -> CaptureManager:
        return self.capture(layer_types=layers)

    def capture(self, layer_types: List[torch.nn.Module] = None, **kwargs) -> CaptureManager:
        for name, layer in flatten_module(self.model):
            if (layer_types is not None) and (isinstance(layer, layer_types) is False):
                continue

            self.capture_layer(name, attach=self._start_attached, **kwargs)
        return self

    def attach(self, *args, **kwargs) -> CaptureManager:
        for layer, capture in self.captures.items():
            capture.attach()

        return self

    def detach(self) -> CaptureManager:
        for capture in self.captures.values():
            if capture:
                capture.remove()

        return self

    def update(self, *args, **kwargs):
        pass

    def batch_update(self, *args, **kwargs):
        pass


class CaptureHandler:
    """base class for capturing intermediate values from pytorch model"""

    def __init__(self, module: nn.Module, keep_input: bool = False, keep_output: bool = False):
        self.keep_input = keep_input
        self.keep_output = keep_output

        self.handle = None

    def __call__(self, module: nn.Module, inputs: torch.Tensor, output: torch.Tensor):
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

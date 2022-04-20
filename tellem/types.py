import torch

from typing import Generic, TypeVar

_USES_TORCH = True

Tensor = TypeVar("Tensor")
Module = TypeVar("Module", torch.nn.Module)
Model = TypeVar("Model")

# PYTORCH SPECIFIC TYPES
Tensor = torch.Tensor
Module = Model = torch.nn.Module

LinearLayer = torch.nn.Linear
ConvLayer = torch.nn.modules.conv._ConvNd

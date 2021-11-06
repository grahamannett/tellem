import torch

from typing import Generic, TypeVar

_USES_TORCH = True

Tensor = TypeVar("Tensor")
RemovableHandle = TypeVar("RemovableHandle")
Module = Layer = TypeVar("Module", "torch.nn.Module", "tensorflow.keras.layers.Layer")
Model = TypeVar("Model")

# PYTORCH SPECIFIC TYPES
Tensor = torch.Tensor
Module = Model = torch.nn.Module

LinearLayer = torch.nn.Linear
# ConvLayer = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]
ConvLayer = torch.nn.modules.conv._ConvNd


# unsure how to do typing correctly for this

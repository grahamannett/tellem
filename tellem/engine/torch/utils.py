import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def flatten_module(module: nn.Module):
    return [[name, layer] for name, layer in module.named_modules() if len(list(layer.named_children())) == 0]


class DataLoaders:
    dataset_types = ["train", "val", "test"]

    def __init__(
        self,
        train: DataLoader = None,
        val: DataLoader = None,
        test: DataLoader = None,
        **kwargs,
    ):
        self.train = train
        self.val = val
        self.test = test

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_dataset(cls, **kwargs):
        obj = {k: v for k, v in kwargs.items() if k in DataLoaders.dataset_types}
        return cls(DataLoader(**obj))

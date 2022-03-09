import torch
import torch.nn as nn

from typing import Callable, List, TypedDict


class DataLoaders:
    dataset_types = ["train", "val", "test"]

    def __init__(
        self,
        train: torch.utils.data.DataLoader = None,
        val: torch.utils.data.DataLoader = None,
        test: torch.utils.data.DataLoader = None,
    ):
        self.train = train
        self.val = val
        self.test = test

    def __getitem__(self, key):
        return self.__dict__[key]

    @classmethod
    def from_dataset(cls, **kwargs):
        obj = {k: v for k, v in kwargs.items() if k in DataLoaders.dataset_types}
        return cls(torch.utils.data.DataLoader())


class Callback:
    def __init__(self, trainer_ref: "TrainerHelper" = None, capture_ref: "CaptureManager" = None) -> None:
        self._trainer_ref = trainer_ref
        self._capture_ref = capture_ref

    def emit(self, *args, **kwargs):
        pass

    @property
    def trainer(self):
        return self._trainer_ref

    @trainer.setter
    def trainer(self, trainer_ref: "TrainerHelper"):
        self._trainer_ref = trainer_ref

    @property
    def capture_manager(self):
        return self._capture_ref

    @capture_manager.setter
    def capture_manager(self, capture_ref: "CaptureManager"):
        self._capture_ref = capture_ref

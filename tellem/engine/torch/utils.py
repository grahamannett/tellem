import torch
import torch.nn as nn

from typing import Callable, List, TypedDict


class DataLoaders(TypedDict):
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


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

import torch
import torch.nn as nn
import torch.optim as optim


import copy

from typing import Callable, Dict, List
from tellem.engine.torch import Capture


from tellem.engine.torch.capture import CaptureManager
from tellem.engine.torch.utils import Callback, DataLoaders


_EmitDefaults = {
    "pre_train_step": [CaptureManager.detach],
    "pre_val_step": [CaptureManager.attach],
    "post_val_step": [CaptureManager.update],
    "val_step_batch_update": [CaptureManager.batch_update],
}


class EmitInfo:
    def __init__(self):
        self.steps = _EmitDefaults
        self._callback_ref = None
        pass

    def __getitem__(self, key):
        return self.steps.get(key, None)

    def __setitem__(self, key, val):
        pass

    def __call__(self, *args, **kwargs):
        pass


class TrainerHelper:
    callbacks: List["Callback"] = []
    capture_manager: CaptureManager = None

    def __init__(
        self,
        model: nn.Module,
        dataloaders: DataLoaders,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.modules.loss._Loss = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = None,
        share_memory: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = {phase: len(dataloader_.dataset) for phase, dataloader_ in dataloaders.items()}

        # defaults
        self._device = device if device is not None else "cpu"
        self.criterion = criterion if criterion is not None else self._default_criterion()
        self.optimizer = optimizer if optimizer is not None else self._default_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._default_scheduler()

        # post-defaults model setup
        if share_memory:
            self.model.share_memory()
        self.model = self.model.to(self._device)

        # tracking
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0
        self.stop_early = False

        self.clear_running_vals()

    def _default_criterion(self):
        return nn.CrossEntropyLoss()

    def _default_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _default_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def _emit(self, step):
        for callback in self.callbacks:
            callback.emit()

    def clear_running_vals(self):
        self.running_loss = 0.0
        self.running_corrects = 0

    def batch_update_loss_and_corrects(
        self, loss: torch.Tensor, inputs: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor
    ) -> None:
        self.running_loss += loss.item() * inputs.size(0)
        self.running_corrects += torch.sum(preds == labels.data)

    def post_step(self, phase: str) -> None:
        self.epoch_loss = self.running_loss / self.dataset_sizes[phase]
        self.epoch_acc = self.running_corrects.double() / self.dataset_sizes[phase]
        print(f"{phase.upper().ljust(10)}| Loss: {self.epoch_loss:.4f} Acc: {self.epoch_acc:.4f}")

    def train_step(self) -> None:
        step = "train"

        self.model.train()
        self.clear_running_vals()
        for inputs, labels in self.dataloaders[step]:
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = self.model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

            self.batch_update_loss_and_corrects(loss, inputs, preds, labels)

        self.scheduler.step()
        self.post_step(step)

    # would be cool to use a @decorator on the batches if i broke that off
    def val_step(self) -> None:
        step = "val"
        self.clear_running_vals()
        self.model.eval()

        # batches of val data
        for inputs, labels in self.dataloaders[step]:
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            self.batch_update_loss_and_corrects(loss, inputs, preds, labels)

            self.emit_info("batch_update")

        # after you are done with all the data for val
        if self.epoch_acc > self.best_acc:
            self.best_acc = self.epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.post_step(step)

    def train_init(self) -> None:
        if self.capture_manager:
            self.capture_manager.init()

    def train(self, num_epochs: int) -> None:

        for epoch in range(num_epochs):
            print(f"{'EPOCH'.ljust(7)}==>{epoch}")

            self.emit_info("pre_train_step")
            self.train_step()
            self.emit_info("post_train_step")

            # val related
            self.emit_info("pre_val_step")
            self.val_step()
            self.emit_info("post_val_step")

            for callback in self.callbacks:
                callback()

            # allow early exit
            if self.stop_early:
                break

    def emit_info(self, step: str):
        pass

    def attach_callback(self, callback: Callback) -> None:
        callback.trainer = self
        callback.capture_manager = self.capture_manager

        self.callbacks.append(callback)

    def attach_capture_manager(self, capture_manager: CaptureManager) -> None:
        self.capture_manager = capture_manager

    def post_val_step(self, *args, **kwargs):
        pass

import torch
import torch.nn as nn
import torch.optim as optim


import copy

from typing import Callable, Dict, List

from tellem.engine.torch import CaptureManager


class TrainerHelper:
    def __init__(self, model: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader], device: str = None) -> None:
        super().__init__()

        self._device = device if device is not None else "cpu"
        self.model = model
        self.model.share_memory()

        self.model = self.model.to(self._device)

        self.dataloaders = dataloaders
        self.dataset_sizes = {phase: len(dataloader_.dataset) for phase, dataloader_ in dataloaders.items()}

        #
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0

        self.capture_manager = None
        self.stop_early = False

        self.callbacks: List[Callback] = []
        self.clear_running_vals()

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

            self.capture_manager.batch_update(labels=labels.detach().cpu().numpy())

        # after you are done with all the data for val
        if self.epoch_acc > self.best_acc:
            self.best_acc = self.epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.post_step(step)

    def train_init(self) -> None:
        if self.capture_manager:
            self.capture_manager.init

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            print(f"{'EPOCH'.ljust(7)}==>{epoch}")

            # train related
            self.capture_manager.detach()
            self.train_step()

            # val related
            self.capture_manager.attach()
            self.val_step()

            # await
            self.capture_manager.update(epoch=epoch)

            for callback in self.callbacks:
                callback()

            # allow early exit
            if self.stop_early:
                break

    def attach_callback(self, callback: Callback) -> None:
        callback.trainer = self
        # callback.set_trainer(self)
        callback.capture_manager = self.capture_manager

        self.callbacks.append(callback)

    def attach_capture_manager(self, capture_manager: "CaptureManager") -> None:
        self.capture_manager = capture_manager

    def post_val_step(self, *args, **kwargs):
        pass


class Callback:
    def __init__(self, trainer_ref: TrainerHelper = None, capture_ref: CaptureManager = None) -> None:
        self.trainer_ref = None
        self.capture_ref = None

    def emit(self, *args, **kwargs):
        pass

    @property
    def trainer(self):
        return self.trainer_ref

    @trainer.setter
    def trainer(self, trainer_ref: TrainerHelper):
        self.trainer_ref = trainer_ref

    @property
    def capture_manager(self):
        return self.capture_ref

    @capture_manager.setter
    def capture_manager(self, capture_ref: CaptureManager):
        self.capture_ref = capture_ref

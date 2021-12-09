import torch
import torch.nn as nn

from typing import Callable, Callback, List


class Callback:
    trainer_ref: "TrainerHelper" = None

    # def __init__(self):
    #     pass
    #     self.trainer_ref = None
    #     self.capture_ref = None

    @property
    def trainer(self):
        return self.trainer_ref

    @trainer.setter
    def trainer(self, trainer_ref: "TrainerHelper"):
        self.trainer_ref = trainer_ref

    @property
    def capture_manager(self):
        return self.capture_ref

    @capture_manager.setter
    def capture_manager(self, capture_ref: "Capture"):
        self.capture_ref = capture_ref


class TrainerHelper:
    """Since pytorch doesnt have anything like the keras equivalent model.fit()
    this is what should be similar.  Has a
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        loss_func: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: str = None,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        self.dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in (("train", self.trainloader), ("test", self.testloader))}

        self.loss_func = loss_func
        self.optimizer = optimizer

        self._capture_manager = None

        self.perturb_helper = None

        self.callbacks: List[Callback] = []
        self.stop_early = False

        self.device = None

    @property
    def capture(self):
        return self._capture_manager

    @capture.setter
    def _(self, capture_manager):
        self._capture_manager = capture_manager

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
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)

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
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)

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
            # asyncio.run(self.capture_manager.update(epoch=epoch))
            # asyncio.run(self.capture_manager.update(epoch=epoch))
            self.capture_manager.update(epoch=epoch)

            for callback in self.callbacks:
                callback()

            # allow early exit
            if self.stop_early:
                break

    def attach_callback(self, callback: "Callback") -> None:
        callback.trainer = self
        # callback.set_trainer(self)
        callback.capture_manager = self.capture_manager

        self.callbacks.append(callback)

    def attach_capture_manager(self, capture_manager: Capture) -> None:
        self.capture_manager = capture_manager

    def post_val_step(self, *args, **kwargs):
        pass

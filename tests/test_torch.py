import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from examples.example_tcav import load_data

from tellem.engine.torch import Capture, CaptureManager
from tellem.engine.torch.train import TrainerHelper, DataLoaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # usually you would use F.relu but
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TestCaptureManager(unittest.TestCase):
    batch_size = 64

    def setUp(self) -> None:

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST("tmp/", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("tmp/", train=False, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.model = Net()

        return super().setUp()

    def test_setup(self):
        x, y = next(iter(self.train_loader))

        self.assertIsNotNone(x)
        self.assertIsNotNone(y)

    def test_model(self):
        x, y = next(iter(self.train_loader))
        outputs = self.model(x)

        self.assertIsNotNone(y)
        self.assertIsNotNone(outputs)

    def test_dataloader(self):
        dataloaders = DataLoaders(train=self.train_loader, test=self.test_loader, val=self.test_loader)
        assert dataloaders["test"] == dataloaders["val"]

    def test_trainer(self):
        dataloaders = DataLoaders(train=self.train_loader, test=self.test_loader, val=self.test_loader)
        trainer = TrainerHelper(self.model, dataloaders=dataloaders)
        trainer.train(1)

    def test_capture(self):
        def hook_fn(grad):
            self.grad__ = grad

        x, y = next(iter(self.train_loader))
        capture = Capture(self.model, "conv1")

        capture.capture_activations()
        preds = self.model(x)
        self.assertEqual(capture.activations.shape, (64, 32, 26, 26))

        preds = self.model(x)
        capture.capture_gradients()
        loss = F.nll_loss(preds, y)
        loss.backward()

        grads = capture.gradients
        self.assertEqual(capture.activations.shape, capture.gradients.shape)

    def test_activations(self):
        x, _ = next(iter(self.train_loader))
        self.capture_manager = CaptureManager(self.model)
        self.capture_manager.capture_layer("conv1")
        self.capture_manager["relu1"] = Capture(self.model, "relu1").capture_activations()

        activations, preds = self.capture_manager(x)

        self.assertEqual(activations["conv1"].shape, (64, 32, 26, 26))
        self.assertEqual(preds.shape, (64, 10))

    @unittest.skip("not implemented")
    def test_gradients(self):
        pass

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# need to fix this import type

from tellem.implementations import TCAV
from tests.common import TestCase
from tests.common.fixtures import ResNetFixture

import pytest


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TestTCAV(TestCase):
    # https://github.com/pytorch/captum/blob/master/tests/attr/test_lime.py

    def setUp(self) -> None:
        self.batch_size = 64
        self.model = Net()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST("tmp/", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("tmp/", train=False, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.x, self.y = next(iter(self.train_loader))

        return super().setUp()

    def test_setup(self):
        tcav = TCAV(self.model)

        self.assertIsNotNone(tcav)

    def test_cav(self):

        tcav = TCAV(self.model)

        self.assertEqual(tcav.cav, {})

        self.assertIsNotNone(tcav)

        tcav.capture_layers("conv1", "conv2")

        concepts = self.x
        non_concepts = torch.rand(self.x.shape)
        tcav.train_cav(concepts, non_concepts)

        y_concepts = F.one_hot(self.y)
        y_non_concepts = torch.zeros_like(y_concepts)

        tcav_scores = tcav.compute_tcav(concepts, y_concepts)

        self.assertNotEqual(tcav_scores, {})

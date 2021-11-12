import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# need to fix this import type

from tellem.implementations import TCAV
from tests.common import TestCase


class TestTCAV(TestCase):
    # https://github.com/pytorch/captum/blob/master/tests/attr/test_lime.py

    def setUp(self) -> None:
        self.batch_size = 64
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST("tmp/", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("tmp/", train=False, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.x, self.y = next(iter(self.train_loader))

        return super().setUp()

    def test_setup(self):
        # x, y = next(iter(self.train_loader))

        tcav = TCAV(self.model)

        self.assertIsNotNone(tcav)

    def test_cav(self):

        tcav = TCAV(self.model)

        self.assertEqual(tcav.cav, {})

        self.assertIsNotNone(tcav)

        tcav.capture_layers("conv1", "relu1")
        # self.assertEqual(len(tcav.capture.activation_layers), 2)

        concepts = self.x
        non_concepts = torch.rand(self.x.shape)
        tcav.train_cav(concepts, non_concepts)

        y_concepts = F.one_hot(self.y)
        y_non_concepts = torch.zeros_like(y_concepts)

        tcav_scores = tcav.test_tcav(concepts, non_concepts, y_concepts, y_non_concepts)

        self.assertNotEqual(tcav_scores, {})

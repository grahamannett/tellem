import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
import torch
import torch.nn as nn

from tellem import backend
from tellem.implementations.base import ImplementationBase

from tellem.types import Tensor, Model


# class Concept(Tensor):
#     def __init__(self) -> None:
#         pass


class Capture:
    def __init__(self, model: nn.Module, layer: str):
        self.model = model
        self.layer = layer
        self._activations = False
        self._gradients = False

        self.activations = None
        self.gradients = None

    def capture_activations(self):
        self._activations = True

        def hook(module, inputs, outputs):
            self.activations = outputs

        module = dict(self.model.named_modules())[self.layer]
        module.register_forward_hook(hook)

    def capture_gradients(self):
        self._gradients = True

        def hook(gradients):
            self.gradients = gradients

        self.activations.register_hook(hook)


class TCAV(ImplementationBase):
    """

    Official Paper:
        - https://arxiv.org/abs/1711.11279
    Official Implementation:
        - https://github.com/tensorflow/tcav

    Args:
        ImplementationBase ([type]): [description]


    Useage:
        model = Model()
        tcav = TCAV(model)
        tcav.use_layers("relu1", "conv2")

        concepts = ...dataloader or tensor of input data for striped images...
        non_concepts = ...tensor of random examples ...

        tcav.train_cav(concepts, non_concepts)

    """

    def __init__(self, model: Model):
        super().__init__()
        self.model = model
        self.cav = {}

    def capture_layers(self, *layers):
        """intermediate layers, they call them bottlenecks"""
        self.capture = {}

        for layer in layers:
            self.capture[layer] = Capture(self.model, layer=layer)
            self.capture[layer].capture_activations()
        # self.captured = CaptureManager(self.model, activation_layers=layers)
        self.cav = {layer: None for layer in layers}

    def train_cav(self, concepts, non_concepts):
        # create the training labels for the linear model
        y_train = torch.stack([torch.ones(len(concepts)), torch.zeros(len(non_concepts))]).reshape(-1)

        # concat the concepts and not so we can generate the activations together
        _ = self.model(torch.cat((concepts, non_concepts), 0))

        for layer in self.capture.keys():
            activations = self.capture[layer].activations
            activations = activations.reshape(len(activations), -1).detach().numpy()
            linear_model = SGDClassifier(loss="hinge", eta0=1, learning_rate="constant", penalty=None)

            linear_model.fit(activations, y_train)
            self.cav[layer] = linear_model.coef_.reshape(-1)

    def test_tcav(self, concepts, non_concepts, y_concepts, y_non_concepts):
        y = torch.vstack([y_concepts, y_non_concepts])
        preds = self.model(torch.cat((concepts, non_concepts), 0))

        for layer in self.capture.keys():
            self.capture[layer].capture_gradients()

        preds.backward(y)

        tcav_scores = {}
        for layer in self.capture.keys():
            gradients = self.capture[layer].gradients
            gradients = gradients.reshape(len(gradients), -1)
            tcav_scores[layer] = gradients @ self.cav[layer]

        return tcav_scores

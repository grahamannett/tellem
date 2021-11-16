import torch
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

from tellem import Capture
from tellem.implementations.base import ImplementationBase
from tellem.types import Model, Tensor

_USES_TORCH = True


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
        tcav.capture_layers("relu1", "conv2")
        tcav.train_cav(concepts, non_concepts)
        tcav_scores = tcav.compute_tcav(concepts, non_concepts)

        concepts = ...dataloader or tensor of input data for striped images...
        non_concepts = ...tensor of random examples ...
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

        self.cav = {layer: None for layer in layers}

    def train_cav(self, concepts: Tensor, non_concepts: Tensor = None):
        """generate the CAV for a layer based on concepts and non-concepts

        Args:
            concepts (Tensor): input data containing a concept
            non_concepts (Tensor): data that is not concept
        """
        # create the training labels for the linear model

        if non_concepts is None:
            non_concepts = torch.rand_like(concepts)

        y_train = torch.stack([torch.ones(len(concepts)), torch.zeros(len(non_concepts))]).reshape(-1)

        # concat the concepts and not so we can generate the activations together
        _ = self.model(torch.cat((concepts, non_concepts), 0))

        for layer in self.capture.keys():
            # for each layer we are 'testing' we get the activations and train a linear classifier, then save the CAV
            activations = self.capture[layer].activations
            activations = activations.reshape(len(activations), -1).detach().numpy()
            linear_model = SGDClassifier(loss="hinge", eta0=1, learning_rate="constant", penalty=None)

            linear_model.fit(activations, y_train)
            self.cav[layer] = linear_model.coef_.reshape(-1)

    def compute_tcav(self, x: Tensor, y: Tensor, **kwargs):
        """[summary]
        Args:
            concepts (Tensor): concepts we are testing the CAV for
            y_concepts (Tensor): labels

        Returns:
            Dict[layer, score]: sensitivity of score to layer
        """

        preds = self.model(x)

        for layer in self.capture.keys():
            self.capture[layer].capture_gradients()

        preds.backward(y)

        cav_sensitivity_scores = {}
        for layer in self.capture.keys():
            grad = self.capture[layer].grad
            grad = grad.reshape(len(grad), -1)
            cav_sensitivity_scores[layer] = grad @ self.cav[layer]

        tcav_scores = {}
        for layer, scores in cav_sensitivity_scores.items():
            tcav_scores[layer] = sum(scores > 0).item() / len(scores)

        return tcav_scores

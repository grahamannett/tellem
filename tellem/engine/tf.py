from tellem.types import Tensor

# from sklearn import SGDClassifier
import tensorflow as tf
import numpy as np

from functools import singledispatchmethod
from typing import Any, Callable, Dict, List, Union

# https://github.com/tensorflow/tensorflow/issues/33478
# https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-843720638
# https://keras.io/examples/vision/grad_cam/
def ____get_layer_output(model, idx, x):
    OutFunc = tf.keras.backend.function([model.input], [model.layers[idx].output])
    return OutFunc(np.expand_dims(x, axis=0))[0]


class ModelWrapper:
    """wrapper around keras/tensorflow class"""

    def __init__(self, model: tf.keras.Model, outputs=None):
        self.model = model

        if outputs is None:
            self.outputs = []
            self.outputs.append(model.outputs)

        self.outputs.append(outputs)

    def __call__(self, *args, **kwargs):

        outputs = self.model._call(*args)
        return self.model(args)

    def gradients(self, x, y):
        pass

    def _has_wrapper(self):
        if hasattr(self.model, "_wrapped"):
            return
        # else:
        #     self._create_wrapper()

    @staticmethod
    def wrap(model: tf.keras.Model, outputs=None):
        """monkey patch the __call__ method

        Args:
            model (tf.keras.Model): [description]
        """
        model._call = model.__call__
        model._wrap = ModelWrapper(model)

        def _call(*args, **kwargs):
            return model._call(*args, *kwargs)

        model.__call__ = _call
        model._add_wrap_layer = ModelWrapper._add_wrap_layer

    @staticmethod
    def _add_wrap_layer(layer):
        pass


class Capture:
    def __init__(self, model: tf.keras.Model, layer: Union[str, int]):
        if hasattr(model, "_has_wrapper"):
            # we dont need to wrap and just add capture layer
            model._add_wrap_layer(layer)
        else:
            ModelWrapper.wrap(model)
            model._add_wrap_layer(layer)

        if not hasattr(model, "_captures"):
            model._captures = []
        model._captures.append(self)

        self.model = model

        self._layer_id = layer
        self.layer = self.get_layer(layer)
        self._activations = False
        self._gradients = False

        self.activations = None
        self.gradients = None

        # if model._hooks:
        #     self.hooks = model._hooks
        self.hooks = []

    @singledispatchmethod
    def get_layer(self, layer):
        raise NotImplementedError("Cannot get layer")

    @get_layer.register
    def _(self, layer: int):
        return self.model.layers[layer]

    @get_layer.register
    def _(self, layer: str):
        return self.model.get_layer(name=layer)

    def _default_gradients_hook(self, grad):
        self.grad = grad

    def capture_activations(self, hook=None):

        self.layer._call_fn = self.layer.call

        def _hook(inputs: tf.Tensor):
            output = self.layer._call_fn(inputs)
            self.activations = output
            return output

        self.hooks.append(_hook)

        self.layer.call = _hook

        # activations = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(self.layer).output, self.model.output])
        return self

    def capture_gradients(self, hook: Callable[..., Any] = None):
        raise NotImplementedError()

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

        # gradient = tf.keras.backend.gradients(model.output, self.layer.input)

        # self.layer._call_fn = self.layer.call

        # with tf.GradientTape() as tape:
        #     def _hook(inputs: tf.Tensor):
        #         outputs = self.layer._call_fn(inputs)
        #         tape.watch(outputs)
        #         return outputs
        #     grads = tape.gradient

        # self._gradients = True
        # # hook_ = self.activations.register_hook(hook)
        # # self.hooks.append(hook_)
        # return self

        # self.tape = tf.GradientTape()
        # grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(self._layer_id).output, self.model.output])
        # # last_conv_layer_output, preds = grad_model(img_array)
        # pred_index = tf.argmax(preds[0])
        # class_channel = preds[:, pred_index]

    def get_grads(self, x, y):
        pass

    def __del__(self):
        self.remove()

    def remove(self):
        # for hook in self.hooks:
        self.layer.call = self.layer._call_fn


# class Capture:
#     def __init__(self, model: tf.keras.Model, idxs: List[int]):
#         self.model = model

#         layer_outputs = [model.layers[idx].output for idx in idxs]
#         if model.layers[-1].output not in layer_outputs:
#             layer_outputs.append(model.layers[len(model.layers) - 1].output)

#         self._capture_func = tf.keras.backend.function([model.input], layer_outputs)

#     def __call__(self, x, y) -> Dict[str, Tensor]:

#         outputs = self._capture_func()
#         return self._capture_func()


# BELOW IS OLD IMPLEMENTATION


class TCAV(object):
    """tcav
    init with model
    then use_bottleneck
    then train_cav
    then calculate_sensitivty
    """

    def __init__(self, model, lm=None, bottleneck=None):
        self.model = model
        self.lm = SGDClassifier() if not lm else lm

        if bottleneck:
            self.use_bottleneck(bottleneck)
        else:
            self.bottleneck_layer = None
            self.model_h = None
            self.model_f = None

        # scores
        self.sensitivity = None
        self.labels = None

        # you want to use this on model.predict()  otherwise easy to get oom
        self.batch_size = 32

    def use_bottleneck(self, bottleneck: int):
        """split the model into 2 models based on model and post models for tcav linear model

        Args:
            layer (int): layer to split nn model
        """
        self.model_f, self.model_h = split_model(self.model, bottleneck=bottleneck)
        self.bottleneck_layer = self.model.layers[bottleneck]

    def train_cav(self, concepts, counterexamples):
        # get activations for both
        x = self.model_f.predict(np.concatenate([concepts, counterexamples]), batch_size=self.batch_size)
        x = x.reshape(x.shape[0], -1)

        y = np.repeat([0, 1], [len(concepts), len(counterexamples)])
        logger.info("fitting the linear model")
        # accr
        self.lm.fit(x, y)
        self.coefs = self.lm.coef_
        self.cav = np.transpose(-1 * self.coefs)

    def test_lm(self, x):
        activations = self.model_f.predict(x)
        return self.lm.predict(activations.reshape(len(activations), -1))

    def calculate_sensitivity(self, concepts, concepts_labels, counterexamples, counterexamples_labels):

        activations = np.concatenate(
            [
                self.model_f.predict(concepts, batch_size=self.batch_size),
                self.model_f.predict(counterexamples, batch_size=self.batch_size),
            ]
        )
        labels = np.concatenate([concepts_labels, counterexamples_labels])

        grad_vals = []

        for x, y in zip(activations, labels):
            # predicting on single thing so need single to be 1 batch
            x = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
            y = tf.convert_to_tensor(np.expand_dims(y, axis=0), dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(x)

                y_out = self.model_h(x)
                loss = tf.keras.backend.categorical_crossentropy(y, y_out)

            grad_vals.append(tape.gradient(loss, x).numpy())

        grad_vals = np.array(grad_vals).squeeze()

        self.sensitivity = np.dot(grad_vals.reshape(grad_vals.shape[0], -1), self.cav)
        self.labels = labels
        self.grad_vals = grad_vals

    def sensitivity_score(self):
        """Print the sensitivities in a readable way"""
        num_classes = self.labels.shape[-1]

        sens_for_class_k = {}
        for k in range(0, num_classes):
            class_idxs = np.where(self.labels[:, k] == 1)
            if len(class_idxs[0]) == 0:
                sens_for_class_k[k] = None
            else:
                sens_for_class = self.sensitivity[class_idxs[0]]
                sens_for_class_k[k] = len(sens_for_class[sens_for_class > 0]) / len(sens_for_class)

        return sens_for_class_k

    def find_best_tcav(self, x_concepts, x_concepts_attack, x_concepts_subset, y_concepts_subset, x_concepts_subset_attack):
        """
        i need to probably rename these.  its unclear what everything is
        """
        self.best = {
            "layer": None,
            "score": 0,
        }

        score_per_layer = []

        y_true = np.repeat([0, 1], [len(x_concepts), len(x_concepts_attack)])
        y_true_subset = np.repeat([0, 1], [len(x_concepts_subset), len(x_concepts_subset_attack)])

        for layer_n in range(0, len(self.model.layers) - 1):

            self.use_bottleneck(layer_n)
            self.train_cav(x_concepts, x_concepts_attack)
            self.calculate_sensitivty(x_concepts_subset, y_concepts_subset, x_concepts_subset_attack, y_concepts_subset)

            y_preds = self.test_lm(np.concatenate([x_concepts_subset, x_concepts_subset_attack]))
            score = accuracy_score(y_preds, y_true_subset)

            if score > self.best["score"]:
                logger.info(f"best tcav layer updated using layer: {layer_n}")
                self.best = {"layer": layer_n, "score": score}

        return self.best

import unittest

import numpy as np
from tellem.engine.tf import Capture

import tensorflow as tf


class TestCapture(unittest.TestCase):
    def setUp(self) -> None:

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 1, activation="relu", name="conv1"),
                tf.keras.layers.Conv2D(64, 1, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        # self.x = np.random.randint(0, 256, size=(64, 32, 32, 3)).astype("float32")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
        x_train = np.expand_dims(x_train.astype("float32") / 255, -1)

        self.x, self.y = x_train[0:5], y_train[0:5]
        # self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # self.x_2, self.y_2 = x_train[5:10], y_train[5:10]
        return super().setUp()

    def test_model_working(self):

        preds = self.model(self.x)

        self.assertIsNotNone(self.model)
        self.assertIsNotNone(preds)

    def test_activations(self):
        capture_manager = {}
        capture_manager["conv1"] = Capture(self.model, "conv1").capture_activations()
        # capture_manager["conv2"] = Capture(self.model, 1).capture_activations()

        _ = self.model(self.x)
        activations1 = capture_manager["conv1"].activations.numpy()
        self.assertIsNotNone(activations1)

        _ = self.model(self.x_test[0:5])
        activations2 = capture_manager["conv1"].activations.numpy()
        self.assertFalse(np.array_equal(activations1, activations2))

    @unittest.skip("gradients not implemented for tensorflow yet")
    def test_gradients(self):
        capture_manager = {}
        # capture_manager["conv1"] = Capture(self.model, "conv1").capture_gradients()

        # x = tf.convert_to_tensor(self.x)
        # y = tf.convert_to_tensor(self.y)
        # with tf.GradientTape() as tape:
        #     tape.watch(x)
        #     outputs = self.model(x)



        # breakpoint()
        # tape.watch(self.x)
        # outputs = self.model(self.x)
        # grads = tf.keras.backend.gradients(outputs, self.y)

        # capture_manager["conv1"] = Capture(self.model, "conv1").capture_activations()

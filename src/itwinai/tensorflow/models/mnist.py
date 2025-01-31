# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Roman Machacek
#
# Credit:
# - Roman Machacek <roman.machacek@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


# import tensorflow.keras as keras
from typing import List

import tensorflow as tf


class MNIST_Model(tf.keras.Model):
    def __init__(self, input_shape: List[int] = (28, 28, 1), output_shape: int = 10):
        super().__init__()

        # LeNet5
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=6, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.AveragePooling2D(2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.AveragePooling2D(2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=120, activation="relu"),
                tf.keras.layers.Dense(units=84, activation="relu"),
                tf.keras.layers.Dense(units=10),
            ]
        )

    def call(self, inputs):
        return self.model(inputs)

# import tensorflow.keras as keras
import tensorflow as tf

from typing import List


class MNIST_Model(tf.keras.Model):
    def __init__(
        self,
        input_shape: List[int] = (28, 28, 1),
        output_shape: int = 10
    ):
        super().__init__()

        self.model = tf.keras.Sequential(
            [
                # tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(
                    32, kernel_size=3, activation="relu",
                    input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(2),
                # tf.keras.layers.Conv2D(
                #     64, kernel_size=3, activation="relu"),
                # tf.keras.layers.MaxPooling2D(pool_size=2),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(output_shape)
            ]
        )

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_shape)
        ])

        # LeNet5
        self.model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(filters=6, kernel_size=(
                3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.AveragePooling2D(2),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.AveragePooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=10)
        ])

    def call(self, inputs):
        return self.model(inputs)

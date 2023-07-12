import keras

from typing import List


class MNIST_Model(keras.Model):
    def __init__(
            self,
            input_shape: List[int],
            output_shape: int
    ):
        super().__init__()

        self.model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(output_shape, activation="softmax")
        ]
    )

    def call(self, inputs):
        return self.model(inputs)
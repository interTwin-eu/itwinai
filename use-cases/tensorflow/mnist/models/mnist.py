import keras

from marshmallow_dataclass import dataclass

@dataclass
class ModelConf:
    input_shape:list
    output_shape:int

def mnist_model(config: ModelConf) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=config.input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(config.output_shape, activation="softmax")
        ]
    )


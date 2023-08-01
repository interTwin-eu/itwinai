import tensorflow as tf
import tensorflow.keras as keras

from itwinai.backend.tensorflow.trainer import TensorflowTrainer
from itwinai.backend.components import Logger
from typing import List


class MNISTTrainer(TensorflowTrainer):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        loss: dict,
        optimizer: dict,
        model: keras.Model,
        loggers: List[Logger],
    ):
        # Configurable
        self.loggers = loggers

        super().__init__(
            loss=keras.losses.get(loss),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            optimizer=keras.optimizers.get(optimizer),
            model_func=lambda: model,
            metrics_func=lambda: [],
            strategy=tf.distribute.MirroredStrategy()
        )

    def train(self, data):
        super().train(data)

    def execute(self, data):
        self.train(data[0])

    def setup(self, args):
        pass

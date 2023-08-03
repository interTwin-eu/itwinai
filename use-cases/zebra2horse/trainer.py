import tensorflow as tf
import tensorflow.keras as keras

from itwinai.backend.tensorflow.trainer import TensorflowTrainer
from itwinai.backend.components import Logger
from typing import List


class Zebra2HorseTrainer(TensorflowTrainer):
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            compile_conf: dict,
            model: keras.Model,
            loggers: List[Logger],
    ):
        # Configurable
        self.loggers = loggers

        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            model_func=lambda: keras.models.clone_model(model),
            compile_conf=compile_conf,
            strategy=None
        )

    def train(self, data):
        super().train(data)

    def execute(self, data):
        self.train(data)

    def setup(self, args):
        pass

from typing import List
import tensorflow as tf
import tensorflow.keras as keras

from itwinai.backend.tensorflow.trainer import TensorflowTrainer
from itwinai.backend.loggers import Logger


class Zebra2HorseTrainer(TensorflowTrainer):
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            compile_conf: dict,
            model: dict,
            logger: List[Logger],
    ):
        # Configurable
        self.logger = logger

        # Parse down the optimizers
        for key in compile_conf.keys():
            compile_conf[key] = keras.optimizers.get(compile_conf[key])

        print(model)

        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            model_dict=model,
            compile_conf=compile_conf,
            strategy=tf.distribute.MirroredStrategy()
        )

    def train(self, data):
        super().train(data)

    def execute(self, data):
        self.train(data)

    def setup(self, args):
        pass

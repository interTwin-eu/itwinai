import tensorflow.keras as keras

from itwinai.backend.components import Trainer
from itwinai.backend.components import Logger
from typing import List

class TensorflowTrainer(Trainer):
    def __init__(
        self,
        epochs: int,
        loss: dict,
        optimizer: dict,
        model: keras.Model,
        loggers: List[Logger],
    ):
        # Configurable
        self.loss = keras.losses.get(loss)
        self.optimizer = keras.optimizers.get(optimizer)
        self.epochs = epochs
        self.model = model
        self.loggers = loggers

    def train(self, data):
        x, y = data
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        for logger in self.loggers:
            logger.log(None)
        self.model.fit(x, y, epochs=self.epochs)

    def execute(self, data):
        self.train(data[0])

    def setup(self, args):
        pass

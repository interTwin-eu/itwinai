import tensorflow.keras as keras

# TODO: Solve relative import
import sys
sys.path.append("..")
from components import Trainer

from marshmallow_dataclass import dataclass


@dataclass
class TrainerConf:
    epochs:int
    loss:dict
    optimizer:dict

class TensorflowTrainer(Trainer):
    def __init__(self, model: keras.Model, logger):
        self.trainer = None
        self.model = model
        self.logger = logger

        # Configurable
        self.loss = None
        self.optimizer = None
        self.epochs = None

    def train(self, data):
        x, y = data
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.logger.log()
        self.model.fit(x, y, epochs=self.epochs)

    def execute(self, data):
        self.train(data[0])

    def config(self, config):
        config = TrainerConf.Schema().load(config['TrainerConf'])
        self.loss = keras.losses.get(config.loss['class_name'])
        self.optimizer = keras.optimizers.get(config.optimizer['identifier'])
        self.epochs = config.epochs



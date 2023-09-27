from typing import List, Dict, Tuple, Optional
import tensorflow as tf
import tensorflow.keras as keras

from itwinai.backend.tensorflow.trainer import TensorflowTrainer
from itwinai.backend.loggers import Logger


class Zebra2HorseTrainer(TensorflowTrainer):
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            compile_conf: Dict,
            model: Dict,
            logger: List[Logger],
    ):
        super().__init__()
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

    def train(self, train_dataset, validation_dataset):
        super().train(train_dataset, validation_dataset)

    def execute(
        self,
        train_dataset,
        validation_dataset,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        train_result = self.train(train_dataset, validation_dataset)
        return (train_result,), config

from typing import Dict, List, Optional, Tuple, Any

# from tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.losses import Loss
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from itwinai.backend.tensorflow.trainer import TensorflowTrainer
from itwinai.backend.loggers import Logger


class MNISTTrainer(TensorflowTrainer):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        loss: Dict,  # Type hint prevents jsonargparse from instantiating it
        optimizer: Dict,  # Type hint prevents jsonargparse from instant. it
        model: Dict,  # Type hint prevents jsonargparse from instantiating it
        strategy: Optional[MirroredStrategy] = None,
        logger: Optional[List[Logger]] = None
    ):
        # Configurable
        self.logger = logger if logger is not None else []
        compile_conf = dict(loss=loss, optimizer=optimizer)
        print(f'STRATEGY: {strategy}')
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            model_dict=model,
            compile_conf=compile_conf,
            strategy=strategy
        )

    def train(self, train_dataset, validation_dataset) -> Any:
        return super().train(train_dataset, validation_dataset)

    def execute(
        self,
        train_dataset,
        validation_dataset,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        return (self.train(train_dataset, validation_dataset),), config

    def load_state(self):
        return super().load_state()

    def save_state(self):
        return super().save_state()

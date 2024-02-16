from typing import Dict, List, Optional, Any

# from tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.losses import Loss
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from itwinai.tensorflow.trainer import TensorflowTrainer
from itwinai.loggers import Logger
from itwinai.components import monitor_exec


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
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            model_dict=model,
            compile_conf=dict(loss=loss, optimizer=optimizer),
            strategy=strategy
        )
        self.save_parameters(**self.locals2params(locals()))
        print(f'STRATEGY: {strategy}')
        self.logger = logger if logger is not None else []

    @monitor_exec
    def execute(self, train_dataset, validation_dataset) -> Any:
        return super().execute(train_dataset, validation_dataset)

    def load_state(self):
        return super().load_state()

    def save_state(self):
        return super().save_state()

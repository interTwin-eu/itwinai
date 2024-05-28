from typing import Tuple
from itwinai.components import Trainer, monitor_exec


class GANTrainer(Trainer):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

    @monitor_exec
    def execute(
        self,
        train_dataset: MLDataset,
        validation_dataset: MLDataset,
        test_dataset: MLDataset
    ) -> Tuple[MLDataset, MLDataset, MLDataset, MLModel]:
        pass

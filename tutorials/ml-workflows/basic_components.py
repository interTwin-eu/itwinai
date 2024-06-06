"""
Here we show how to implement component interfaces in a simple way.
"""
from typing import List, Optional, Tuple, Any
from itwinai.components import (
    DataGetter, DataSplitter, Trainer, Saver, monitor_exec
)


class MyDataGetter(DataGetter):
    def __init__(self, data_size: int, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.data_size = data_size
        self.save_parameters(data_size=data_size)

    @monitor_exec
    def execute(self) -> List[int]:
        """Return a list dataset.

        Returns:
            List[int]: dataset
        """
        return list(range(self.data_size))


class MyDatasetSplitter(DataSplitter):
    @monitor_exec
    def execute(
        self,
        dataset: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Splits a list dataset into train, validation and test datasets.

        Args:
            dataset (List[int]): input list dataset.

        Returns:
            Tuple[List[int], List[int], List[int]]: train, validation, and
            test datasets.
        """
        train_n = int(len(dataset)*self.train_proportion)
        valid_n = int(len(dataset)*self.validation_proportion)
        train_set = dataset[:train_n]
        vaild_set = dataset[train_n:train_n+valid_n]
        test_set = dataset[train_n+valid_n:]
        return train_set, vaild_set, test_set


class MyTrainer(Trainer):
    def __init__(self, lr: float = 1e-3, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(name=name, lr=lr)

    @monitor_exec
    def execute(
        self,
        train_set: List[int],
        vaild_set: List[int],
        test_set: List[int]
    ) -> Tuple[List[int], List[int], List[int], str]:
        """Dummy ML trainer mocking a ML training algorithm.

        Args:
            train_set (List[int]): training dataset.
            vaild_set (List[int]): validation dataset.
            test_set (List[int]): test dataset.

        Returns:
            Tuple[List[int], List[int], List[int], str]: train, validation,
            test datasets, and trained model.
        """
        return train_set, vaild_set, test_set, "my_trained_model"


class MySaver(Saver):
    @monitor_exec
    def execute(self, artifact: Any) -> Any:
        """Saves an artifact to disk.

        Args:
            artifact (Any): artifact to save (e.g., dataset, model).

        Returns:
            Any: input artifact.
        """
        return artifact

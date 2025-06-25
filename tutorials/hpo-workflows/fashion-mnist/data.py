# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from itwinai.components import DataGetter, DataSplitter

py_logger = logging.getLogger(__name__)


class FashionMNISTGetter(DataGetter):
    def __init__(self, data_dir: str | Path = "data") -> None:
        super().__init__()
        self.data_dir = data_dir

    def execute(self) -> Dataset:
        """Load the FashionMNIST dataset from the specified directory."""
        py_logger.info("Loading FashionMNIST dataset...")
        train_dataset = datasets.FashionMNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        py_logger.info("Loading complete!")
        return train_dataset


class FashionMNISTSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: float,
        validation_proportion: float,
        test_proportion: float,
        name: str | None = None,
    ) -> None:
        super().__init__(train_proportion, validation_proportion, test_proportion, name)

    def execute(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """Split the dataset into train, validation, and test sets."""
        py_logger.info("Splitting dataset...")
        total_size = len(dataset)
        train_size = int(self.train_proportion * total_size)
        validation_size = int(self.validation_proportion * total_size)
        test_size = total_size - train_size - validation_size

        train_dataset, validation_dataset, test_dataset = random_split(
            dataset, [train_size, validation_size, test_size]
        )
        py_logger.info("Splitting complete!")

        return train_dataset, validation_dataset, test_dataset

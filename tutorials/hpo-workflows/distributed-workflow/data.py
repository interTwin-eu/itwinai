# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from itwinai.components import DataGetter, DataSplitter

data_dir = Path("data")


def download_fashion_mnist() -> None:
    """Download the FashionMNIST dataset using torchvision."""
    print("Downloading FashionMNIST dataset...")
    datasets.FashionMNIST(
        data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    datasets.FashionMNIST(
        data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    print("Download complete!")


class FashionMNISTGetter(DataGetter):
    def __init__(self) -> None:
        super().__init__()

    def execute(self) -> Tuple[Dataset, Dataset]:
        """Load the FashionMNIST dataset from the specified directory."""
        print("Loading FashionMNIST dataset...")
        train_dataset = datasets.FashionMNIST(
            data_dir,
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        print("Loading complete!")
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
        print("Splitting dataset...")
        total_size = len(dataset)
        train_size = int(self.train_proportion * total_size)
        validation_size = int(self.validation_proportion * total_size)
        test_size = total_size - train_size - validation_size

        train_dataset, validation_dataset, test_dataset = random_split(
            dataset, [train_size, validation_size, test_size]
        )
        print("Splitting complete!")
        return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FashionMNIST Dataset Loader")
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download the FashionMNIST dataset and exit",
    )
    args = parser.parse_args()

    if args.download_only:
        download_fashion_mnist()
        sys.exit()

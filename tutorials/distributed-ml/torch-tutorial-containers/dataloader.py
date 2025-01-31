# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Dataloader for Torch-based MNIST use case."""

import os
import shutil
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from itwinai.components import DataGetter, monitor_exec


class MNISTDataModuleTorch(DataGetter):
    """Download MNIST dataset for torch."""

    def __init__(
        self,
        save_path: str = ".tmp/",
    ) -> None:
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.save_path = save_path

    @monitor_exec
    def execute(self) -> Tuple[Dataset, Dataset]:
        train_dataset = datasets.MNIST(
            self.save_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        validation_dataset = datasets.MNIST(
            self.save_path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        print("Train and validation datasets loaded.")
        return train_dataset, validation_dataset, None


class InferenceMNIST(Dataset):
    """Loads a set of MNIST images from a folder of JPG files."""

    def __init__(
        self, root: str, transform: Optional[Callable] = None, supported_format: str = ".jpg"
    ) -> None:
        self.root = root
        self.transform = transform
        self.supported_format = supported_format
        self.data = dict()
        self._load()

    def _load(self):
        for img_file in os.listdir(self.root):
            if not img_file.lower().endswith(self.supported_format):
                continue
            filename = os.path.basename(img_file)
            img = Image.open(os.path.join(self.root, img_file))
            self.data[filename] = img

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_identifier, image) where image_identifier
                is the unique identifier for the image (e.g., filename).
        """
        img_id, img = list(self.data.items())[index]

        if self.transform is not None:
            img = self.transform(img)

        return img_id, img

    @staticmethod
    def generate_jpg_sample(root: str, max_items: int = 100):
        """Generate a sample dataset of JPG images starting from
            LeCun's test dataset.

        Args:
            root (str): sample path on disk
            max_items (int, optional): max number of images to
                generate. Defaults to 100.
        """
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root)

        test_data = datasets.MNIST(root=".tmp", train=False, download=True)
        for idx, (img, _) in enumerate(test_data):
            if idx >= max_items:
                break
            savepath = os.path.join(root, f"digit_{idx}.jpg")
            img.save(savepath)


class MNISTPredictLoader(DataGetter):
    def __init__(self, test_data_path: str) -> None:
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.test_data_path = test_data_path

    @monitor_exec
    def execute(self) -> Dataset:
        return InferenceMNIST(
            root=self.test_data_path,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

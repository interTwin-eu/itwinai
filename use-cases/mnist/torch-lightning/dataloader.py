# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from itwinai.components import DataGetter, monitor_exec


class LightningMNISTDownloader(DataGetter):
    def __init__(self, data_path: str, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.data_path = data_path
        self._downloader = MNISTDataModule(
            data_path=self.data_path,
            download=True,
            # Mock other args...
            batch_size=1,
            train_prop=0.5,
        )

    @monitor_exec
    def execute(self) -> None:
        # Simulate dataset creation to force data download
        self._downloader.setup(stage="fit")
        self._downloader.setup(stage="test")
        self._downloader.setup(stage="predict")


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self, data_path: str, batch_size: int, train_prop: float, download: bool = True
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.download = download
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit":
            mnist_full = MNIST(
                self.data_path, train=True, download=self.download, transform=self.transform
            )
            n_train_samples = int(self.train_prop * len(mnist_full))
            n_val_samples = len(mnist_full) - n_train_samples
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [n_train_samples, n_val_samples]
            )

        if stage == "test":
            self.mnist_test = MNIST(
                self.data_path, train=False, download=self.download, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_path, train=False, download=self.download, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=4)

"""Dataloader for Torch-based MNIST use case."""

from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import transforms, datasets

from itwinai.components import DataGetter


class MNISTDataModuleTorch(DataGetter):
    """Download MNIST dataset for torch."""

    def __init__(
            self,
            save_path: str = '.tmp/',
            # batch_size: int = 32,
            # pin_memory: bool = True,
            # num_workers: int = 4
    ) -> None:
        super().__init__()
        self.save_path = save_path
        # self.batch_size = batch_size
        # self.pin_memory = pin_memory
        # self.num_workers = num_workers

    def load(self):
        self.train_dataset = datasets.MNIST(
            self.save_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.val_dataset = datasets.MNIST(
            self.save_path, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Tuple[Dataset, Dataset], Optional[Dict]]:
        self.load()
        print("Train and valid datasets loaded.")
        # train_dataloder = DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     pin_memory=self.pin_memory,
        #     num_workers=self.num_workers
        # )
        # validation_dataloader = DataLoader(
        #     self.val_dataset,
        #     batch_size=self.batch_size,
        #     pin_memory=self.pin_memory,
        #     num_workers=self.num_workers
        # )
        # return (train_dataloder, validation_dataloader)
        return (self.train_dataset, self.val_dataset), config

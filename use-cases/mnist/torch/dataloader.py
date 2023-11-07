"""Dataloader for Torch-based MNIST use case."""

from typing import Dict, Optional, Tuple, Callable, Any
import os
import shutil

from PIL import Image
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


class InferenceMNIST(Dataset):
    """Loads a set of MNIST images from a folder of JPG files."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        supported_format: str = '.jpg'
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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        # img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img_id, img

    @staticmethod
    def generate_jpg_sample(
        root: str,
        max_items: int = 100
    ):
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

        test_data = datasets.MNIST(root='.tmp', train=False, download=True)
        for idx, (img, _) in enumerate(test_data):
            if idx >= max_items:
                break
            savepath = os.path.join(root, f'digit_{idx}.jpg')
            img.save(savepath)


class MNISTPredictLoader(DataGetter):
    def __init__(
        self,
        test_data_path: str
    ) -> None:
        super().__init__()
        self.test_data_path = test_data_path

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Tuple[Dataset, Dataset], Optional[Dict]]:
        data = self.load()
        return data, config

    def load(self) -> Dataset:
        return InferenceMNIST(
            root=self.test_data_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

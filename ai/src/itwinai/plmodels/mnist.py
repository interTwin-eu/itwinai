"""
Pytorch Lightning models for MNIST dataset.
"""

from typing import Callable, List, Optional, Union
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from .base import ItwinaiBasePlModule, ItwinaiBasePlDataModule


class MNISTDataModule(ItwinaiBasePlDataModule):
    """Pytorch Lightning data module for MNIST dataset

        Args:
            data_dir (str): path to dataset directory
            batch_size (int, optional): batch size. Defaults to 32.
            train_prop (float, optional): proportion of examples in the train
                split, after dataset is split into train and validation.
                Defaults to 0.7.
            transform (Optional[Callable], optional): transformations to apply
                to the loaded images. Defaults to None.
    """

    def __init__(
        self,
            data_dir: str,
            batch_size: int = 32,
            train_prop: float = 0.7,
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_prop = train_prop
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            self.transform = transform

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True,
                transform=self.transform,
                download=False
            )
            n_train_samples = int(self.train_prop * len(mnist_full))
            n_val_samples = len(mnist_full) - n_train_samples
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [n_train_samples, n_val_samples]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                download=False,
                transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir,
                train=False,
                download=False,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size
        )

    def preds_to_names(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[str]:
        """Convert predictions to class names."""
        # Convert prediction to label: in this case is easy, as the label
        # is an integer.
        if not isinstance(preds, list):
            preds = [preds]

        names = []
        for p in preds:
            p += 1
            names.extend([str(el) for el in p.tolist()])
        return names


class LitMNIST(ItwinaiBasePlModule):
    """
    Simple PL model for MNIST.
    Adapted from
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
    """

    def __init__(
        self,
        hidden_size: int = 64,
    ):
        super().__init__()

        # Automatically save constructor args as hyperparameters
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.hidden_size = hidden_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log metrics with autolog
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True
        )

        # Good alternative
        # # Log with generic logger
        # self.logger.log_metrics(
        #     metrics=dict(train_loss=loss.item()),
        #     step=self.global_step
        # )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # good alternative
        # # Log with generic logger
        # self.logger.log_metrics(
        #     metrics=dict(val_loss=loss.item()),
        #     step=self.global_step
        # )
        # self.logger.log_metrics(
        #     metrics=dict(val_acc=acc.item()),
        #     step=self.global_step
        # )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

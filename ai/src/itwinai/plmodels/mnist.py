"""
Pytorch Lightning models for MNIST dataset.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from .base import ItwinaiBasePlModel


class LitMNIST(ItwinaiBasePlModel):
    """
    Simple PL model for MNIST.
    Adapted from
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
    """

    def __init__(
        self,
        data_dir: str,
        hidden_size: int = 64,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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

    # def on_train_start(self):
    #     # Save hyperparameters (alternative way)
    #     self.logger.log_hyperparams(
    #         dict(
    #             data_dir=self.data_dir,
    #             hidden_size=self.hidden_size,
    #             learning_rate=self.learning_rate,
    #             batch_size=self.batch_size
    #         )
    #     )

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    # def prepare_data(self):
    #     MNIST(self.data_dir, train=True, download=False)
    #     MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True,
                transform=self.transform,
                download=False
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
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

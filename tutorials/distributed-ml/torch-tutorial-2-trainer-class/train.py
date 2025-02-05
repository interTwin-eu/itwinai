# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision import datasets, transforms

from itwinai.loggers import MLFlowLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer


# Step 1: setup your neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    # Step 2 (optional): Parse your arguments from the command line
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=14, help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--strategy", type=str, default="ddp", help="distributed strategy (default=ddp)"
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="learning rate (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    # Step 3: Create your datasets
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST("../data", train=False, transform=transform)

    # Step 4: Configure your model and your training configuration
    model = Net()

    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        optim_lr=args.lr,
        optimizer="adadelta",
        loss="cross_entropy",
    )

    # Step 5 (optional): Configure a logger and some metrics
    logger = MLFlowLogger(experiment_name="mnist-tutorial", log_freq=10)

    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=10),
    }

    # Step 6: Create your Trainer
    trainer = TorchTrainer(
        config=training_config,
        model=model,
        metrics=metrics,
        logger=logger,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        checkpoint_every=args.ckpt_interval,
    )

    # Step 7: Launch your training
    _, _, _, trained_model = trainer.execute(train_dataset, validation_dataset, None)


if __name__ == "__main__":
    main()

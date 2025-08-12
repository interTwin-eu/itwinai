# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------
from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from itwinai.constants import EPOCH_TIME_DIR
from itwinai.parser import ArgumentParser as ItwinaiArgParser


def imagenet_dataset(data_root: str, subset_size: int | None = None):
    """Create a torch dataset object for Imagenet."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    imagenet = datasets.ImageFolder(root=data_root, transform=transform)

    if subset_size is None:
        # We do this because we always want to return an instance of a subset, to make
        # everything as consistent as possible
        subset_size = len(imagenet)
    if subset_size > len(imagenet):
        raise ValueError("Limit higher than the total length of the dataset")

    return Subset(imagenet, range(subset_size))


def train_epoch(
    model: nn.Module,
    device: device,
    train_loader: DataLoader,
    optimizer: Optimizer,
):
    """Train a pytorch model for a single epoch with the given arguments."""

    total_loss = 0
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def get_parser() -> ItwinaiArgParser:
    parser = ItwinaiArgParser(description="PyTorch Imagenet scaling test")

    parser.add_argument(
        "--data-dir",
        default="./",
        help=("location of the training dataset in the local filesystem"),
    )
    parser.add_argument(
        "--log-int",
        type=int,
        default=10,
        help="log interval per training. Disabled if < 0.",
    )
    parser.add_argument(
        "--nworker",
        type=int,
        default=0,
        help=("number of workers in DataLoader (default: 0 - only main)"),
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="prefetch data in DataLoader (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="momentum in SGD optimizer (default: 0.5)",
    )
    parser.add_argument(
        "--shuff",
        action="store_true",
        default=False,
        help="shuffle dataset (default: False)",
    )
    parser.add_argument(
        "--rnd-seed",
        type=int,
        default=None,
        help="seed integer for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="backend for parallelisation (default: nccl)",
    )
    parser.add_argument(
        "--subset-size",
        type=Union[int, None],
        default=None,
        help="How big of a subset of ImageNet to use during training.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables GPGPUs"
    )
    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce",
    )
    parser.add_argument(
        "--use-adasum",
        action="store_true",
        default=False,
        help="use adasum algorithm to do reduction",
    )
    parser.add_argument(
        "--gradient-predivide-factor",
        type=float,
        default=1.0,
        help="apply gradient pre-divide factor in optimizer (default: 1.0)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["ddp", "horovod", "deepspeed"],
        default="ddp",
    )
    parser.add_argument(
        "--epoch-time-directory",
        type=str,
        default=f"scalability-metrics/{EPOCH_TIME_DIR}",
        help="Where to store the epoch time metrics used in the scalability report",
    )
    return parser

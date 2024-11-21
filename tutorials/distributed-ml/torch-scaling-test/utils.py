import argparse
import torch.nn as nn
from itwinai.parser import ArgumentParser as ItwinaiArgParser
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.optimizer import Optimizer
from torch import device


def imagenet_dataset(data_root: str):
    """Create a torch dataset object for Imagenet."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    imagenet = datasets.ImageFolder(root=data_root, transform=transform)
    limit = 2000
    imagenet = Subset(imagenet, range(limit))
    return imagenet


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


def parse_params():
    parser = ItwinaiArgParser(description="PyTorch Imagenet scaling test")

    parser.add_argument(
        "--data-dir",
        default="./",
        help=("location of the training dataset in the " "local filesystem"),
    )
    parser.add_argument(
        "--log-int",
        type=int,
        default=10,
        help="log interval per training. Disabled if < 0.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Print parsed arguments",
    )
    parser.add_argument(
        "--nworker",
        type=int,
        default=0,
        help=("number of workers in DataLoader " "(default: 0 - only main)"),
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
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
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
        "--no-cuda", action="store_true", default=False, help="disables GPGPUs"
    )

    args = parser.parse_args()

    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", "\n".join(args_list))
    return args

from typing import Optional
import numpy as np
import random

import torch
from torchvision import datasets, transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(rnd_seed: Optional[int], use_cuda: bool) -> torch.Generator:
    """Set torch random seed and return a PRNG object.

    Args:
        rnd_seed (Optional[int]): random seed. If None, the seed is not set.
        use_cuda (bool): whether GPU is available. 

    Returns:
        torch.Generator: PRNG object.
    """
    g = torch.Generator()
    if rnd_seed is not None:
        # Deterministic execution
        torch.manual_seed(rnd_seed)
        g.manual_seed(rnd_seed)
        if use_cuda:
            torch.cuda.manual_seed(rnd_seed)
    return g


def imagenet_dataset(data_root: str):
    """Create a torch dataset object for Imagenet."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imagenet = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )
    return imagenet

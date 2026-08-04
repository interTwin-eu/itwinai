# --------------------------------------------------------------------------------------
# Part of the RI-SCALE Project: https://www.riscale.eu/
# --------------------------------------------------------------------------------------
"""Synthetic model and datasets for the Model Hub push/pull tutorial."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SanityCheckModel(nn.Module):
    """Small 3-layer CNN used throughout this tutorial."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


def sanity_check_model_class():
    """Returns the (uninstantiated) `SanityCheckModel` class.

    Used as a `_target_` in config.yaml when a config value needs to be a *class*
    rather than an instance -- e.g. in the `ModelHubModelLoader.model_class`.
    """
    return SanityCheckModel


class SyntheticCheckpointDataset(Dataset):
    """Random (input, target) pairs used to train/validate `SanityCheckModel`."""

    def __init__(self, num_samples: int = 64):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(2, 32, 32)
        y = torch.randn(1, 32, 32)
        return x, y


class SyntheticInferenceDataset(Dataset):
    """Random inputs (no targets), each paired with an item ID."""

    def __init__(self, num_samples: int = 16):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(2, 32, 32)
        return idx, x

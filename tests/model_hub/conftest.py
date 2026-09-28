import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SanityCheckModel(nn.Module):
    """Example model architecture to test feature."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


class SyntheticInferenceDataset(Dataset):
    def __init__(self, n=4, h=16, w=16):
        self.n, self.h, self.w = n, h, w

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"sample_{idx}", torch.randn(2, self.h, self.w)


@pytest.fixture
def sanity_check_model_class():
    return SanityCheckModel


@pytest.fixture
def synthetic_inference_dataset_class():
    return SyntheticInferenceDataset

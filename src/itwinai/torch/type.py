"""Custom types definition."""

from typing import Callable

import torch
from torch import nn

#: Torch Loss function.
Loss = nn.Module

#: Torch learning rate scheduler
LrScheduler = nn.Module

#: Torch data batch sampled by a ``DataLoader``.
Batch = torch.Tensor

#: Torch metric function provided by ``torchmetrics`` library.
Metric = Callable


class UninitializedStrategyError(Exception):
    """Error raised when a strategy has not been initialized."""


class DistributedStrategyError(Exception):
    """Error raised when a strategy has already been initialized."""

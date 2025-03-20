# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Custom types definition."""

from typing import Callable

import torch

#: Torch data batch sampled by a ``DataLoader``.
Batch = torch.Tensor

#: Torch metric function provided by ``torchmetrics`` library.
Metric = Callable


class UninitializedStrategyError(Exception):
    """Error raised when a strategy has not been initialized."""


class DistributedStrategyError(Exception):
    """Error raised when a strategy has already been initialized."""

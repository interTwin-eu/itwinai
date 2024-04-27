"""Custom types definition."""

from typing import Callable
from enum import Enum, EnumMeta

import torch
from torch import nn

Loss = nn.Module
LrScheduler = nn.Module
Batch = torch.Tensor
Metric = Callable


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TorchDistributedBackend(BaseEnum):
    """
    Enum for torch distributed backends.
    Reference: https://pytorch.org/docs/stable/distributed.html#backends
    """
    DEFAULT = 'nccl'
    GLOO = 'gloo'
    NCCL = 'nccl'
    MPI = 'mpi'


class TorchDistributedStrategy(BaseEnum):
    DEFAULT = None
    NONE = None
    DDP = 'ddp'
    HVD = 'horovod'
    DS = 'deepspeed'


class TorchLoss(BaseEnum):
    """
    Torch loss class names.
    TODO: complete from https://pytorch.org/docs/stable/nn.html#loss-functions
    """
    L1 = 'L1Loss'
    MSE = 'MSELoss'
    CROSS_ENTROPY = 'CrossEntropyLoss'
    NLLLOSS = 'NLLLoss'


class TorchOptimizer(BaseEnum):
    """
    Torch optimizer class names.
    TODO: complete from https://pytorch.org/docs/stable/optim.html#algorithms
    """
    SGD = 'SGD'
    ADAM = 'Adam'


class UninitializedStrategyError(Exception):
    """Error raised when a strategy has not been initialized."""

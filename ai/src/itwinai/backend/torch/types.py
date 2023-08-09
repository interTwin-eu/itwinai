from enum import Enum, EnumMeta


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
    GLOO = 'gloo'
    NCCL = 'nccl'
    MPI = 'mpi'


class TorchDistributedStrategy(BaseEnum):
    NONE = None
    DDP = 'ddp'


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

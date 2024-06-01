"""Default configuration"""

from pydantic import BaseModel


class Configuration(BaseModel, extra='allow'):
    """Base configuration class."""

    def __getitem__(self, idx):
        return self.__getattribute__(idx)


class TrainingConfiguration(Configuration):
    """Default configuration object for training.
    Override and/or create new configurations using the constructor.

    Example:

    >>> cfg = TrainingConfiguration(batch_size=2, param_a=42)
    >>> print(cfg.batch_size)  # returns 17 (overrides default)
    >>> print(cfg.param_a)     # returns 42 (new value)
    >>> print(cfg.pin_memory)  # returns the default value
    >>>
    >>> from rich import print
    >>> print(cfg)             # pretty-print of configuration

    """
    #: Batch size. In a distributed environment it is usually the
    #: per-worker batch size. Defaults to 32.
    batch_size: int = 32
    #: Whether to pin GPU memory. Property of torch ``DataLoader``.
    #: Defaults to False.
    pin_memory: bool = False
    #: Number of parallel workers used by torch ``DataLoader``.
    #: Defaults to 4.
    num_workers: int = 4
    #: Learning rate used by the optimizer. Defaults to 1e-3.
    lr: float = 1e-3
    #: Momentum used by some optimizers (e.g., SGD). Defaults to 0.9.
    momentum: float = .9
    #: Parameter of Horovod's ``DistributedOptimizer``: uses float16
    #: operations in the allreduce
    #: distributed gradients aggregation. Better performances at
    #: lower precision. Defaults to False.
    fp16_allreduce: bool = False
    #: Parameter of Horovod's ``DistributedOptimizer``: use Adasum
    #: optimization.
    #: Defaults to False.
    use_adasum: bool = False
    #: Parameter of Horovod's ``DistributedOptimizer``: scale
    #: gradients before adding them up.
    #: Defaults to 1.0.
    gradient_predivide_factor: float = 1.0

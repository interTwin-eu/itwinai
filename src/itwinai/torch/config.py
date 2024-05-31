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
    # DataLoader
    batch_size: int = 32
    pin_memory: bool = False
    num_workers: int = 4

    # Optimization
    lr: float = 1e-3
    momentum: float = .9

    # Horovod
    fp16_allreduce: bool = False
    use_adasum: bool = False
    gradient_predivide_factor: float = 1.0

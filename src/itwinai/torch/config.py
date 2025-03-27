# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Default configuration"""

from typing import Any, Dict, Iterable, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field


class Configuration(BaseModel):
    """Base configuration class."""

    __pydantic_extra__: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

    def __getitem__(self, idx):
        return self.__getattribute__(idx)


class TrainingConfiguration(Configuration):
    """Default configuration object for training.
    Override and/or create new configurations using the constructor.

    Example:

    >>> cfg = TrainingConfiguration(batch_size=17, param_a=42)
    >>> print(cfg.batch_size)  # returns 17 (overrides default)
    >>> print(cfg.param_a)     # returns 42 (new value)
    >>> print(cfg.pin_memory)  # returns the default value
    >>>
    >>> from rich import print
    >>> print(cfg)             # pretty-print of configuration

    .. warning::

        Don't reinvent parameters that already exist in the training coniguration, if possible.
        Instead, use the name of the parameters in the training configuration when possible to
        avoid inconsistencies. For instance, the training configuration defines the learning
        rate as ``optim_lr``, so if you redefine it as ``lr`` by doing
        ``TrainingConfiguration(lr=0.005)`` in the configuration you will now have both
        ``optim_lr`` (created by default) and ``lr`` (created by you). This may create
        confusion and potentially (and silently) break the logic in your code.
    """

    #: Batch size. In a distributed environment it is usually the
    #: per-worker batch size. Defaults to 32.
    batch_size: int = 32
    #: Whether to shuffle train dataset when creating a torch ``DataLoader``.
    #: Defaults to False.
    shuffle_train: bool = False
    #: Whether to shuffle validation dataset when creating a torch
    #: ``DataLoader``.
    #: Defaults to False.
    shuffle_validation: bool = False
    #: Whether to shuffle test dataset when creating a torch ``DataLoader``.
    #: Defaults to False.
    shuffle_test: bool = False
    #: Whether to pin GPU memory. Property of torch ``DataLoader``.
    #: Defaults to False.
    pin_gpu_memory: bool = False
    #: Number of parallel workers used by torch ``DataLoader``.
    #: Defaults to 4.
    num_workers_dataloader: int = 4
    #: Loss function. Defaults to 'cross_entropy'
    loss: Literal["mse", "nllloss", "cross_entropy", "l1", "l2", "bceloss"] = "cross_entropy"
    #: Name of the optimizer to use. Defaults to 'adam'.
    optimizer: Literal["adadelta", "adam", "adamw", "rmsprop", "sgd"] = "adam"
    #: Learning rate used by the optimizer. Defaults to 1e-3.
    optim_lr: float = 1e-3
    #: Momentum used by some optimizers (e.g., SGD). Defaults to 0.9.
    optim_momentum: float = 0.9
    #: Betas of Adam optimizer (if used). Defaults to (0.9, 0.999).
    optim_betas: Tuple[float, float] = (0.9, 0.999)
    #: Weight decay parameter for the optimizer. Defaults to 0.
    optim_weight_decay: float = 0.0
    #: Learning rate scheduler algorithm. Defaults to None (not used).
    lr_scheduler: (
        Literal["step", "multistep", "constant", "linear", "exponential", "polynomial"] | None
    ) = None
    #: Learning rate scheduler step size, if needed by the scheduler. Defaults to 10 (epochs).
    lr_scheduler_step_size: int | Iterable[int] = 10
    #: Learning rate scheduler step size, if needed by the scheduler.
    #: Usually this is used by the ExponentialLR.
    # : Defaults to 0.5.
    lr_scheduler_gamma: float = 0.95
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

    # TODO: move this inside the some "run" or "scaling" dedicated config
    #: Torch distributed
    #: `backend <https://pytorch.org/docs/stable/distributed.html#backends>`_.
    #: Defaults to ``nccl``.
    dist_backend: Literal["nccl", "gloo", "mpi"] = "nccl"

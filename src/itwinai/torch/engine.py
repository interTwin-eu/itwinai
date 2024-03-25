"""
Model engine which wraps a torch NN. Still under development. May be removed...
"""

import abc
from typing import Any, Union, Optional, Callable

from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda import amp
from torch import autocast


class OptimizerConfig:
    def __init__(self, optim_class, **kwargs) -> None:
        self.optim_class = optim_class
        self.kwargs = kwargs

    def to_optim(self, parameters) -> optim.Optimizer:
        return self.optim_class(parameters, **self.kwargs)


class LRSchedulerConfig:
    def __init__(self, scheduler_class, **kwargs) -> None:
        self.scheduler_class = scheduler_class
        self.kwargs = kwargs

    def to_scheduler(self, optim) -> LRScheduler:
        return self.scheduler_class(optim, **self.kwargs)


class ModelEngineConfig(BaseModel):
    mixed_precision: bool = False


class ModelEngine(abc.ABC):
    """Wrapper around ML model, which abstracts from distributed and
    mixed-precision models.
    """

    model: nn.Module
    _model_parameters: Any
    optimizer: optim.Optimizer
    lr_scheduler: LRScheduler
    # config: ModelEngineConfig
    mixed_precision: bool = False
    grad_scaler: amp.GradScaler = None

    def __init__(
        self,
        model: nn.Module,
        # model_parameters: Any,
        optimizer: Union[optim.Optimizer, OptimizerConfig],
        lr_scheduler: Optional[Union[LRScheduler, LRSchedulerConfig]] = None,
        mixed_precision: bool = False
        # config: Optional[ModelEngineConfig] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # self._model_parameters = model_parameters
        # if isinstance(optimizer, OptimizerConfig):
        #     self.optimizer = optimizer.to_optim(model_parameters)
        # else:
        #     self.optimizer = optimizer

        # if isinstance(lr_scheduler, LRSchedulerConfig):
        #     self.lr_scheduler = lr_scheduler.to_scheduler(self.optimizer)
        # else:
        #     self.lr_scheduler = lr_scheduler

        # if not config:
        #     self.config = ModelEngineConfig()
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.grad_scaler = amp.GradScaler()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Performs the forward operation."""
        # Wrapper of self.forward()
        return self.forward(*args, **kwds)

    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Performs the forward operation."""
        return self.model(*args, **kwds)

    def train(self, mode: bool = True) -> nn.Module:
        """Set model in training mode."""
        self.model.train(mode=mode)
        return self.model

    def eval(self) -> nn.Module:
        """Set model in inference mode."""
        self.model.eval()
        return self.model

    def to(self, device) -> nn.Module:
        """Move model to specified device."""
        self.model.to(device)
        return self.model

    @abc.abstractmethod
    def zero_grad():
        """Set gradients to zero for the optimizer."""

    @abc.abstractmethod
    def backward(self, loss_fn: Callable, *loss_args) -> torch.Tensor:
        """Perform backward pass and return the loss.

        Args:
            loss_fn (Callable): computes the loss.
            *loss_args: are the arguments to be passed to ``loss_fn``.

        Returns:
            torch.Tensor: computed loss.
        """

    @abc.abstractmethod
    def optimizer_step(self):
        """Perform optimizer step."""

    @abc.abstractmethod
    def lr_scheduler_step(self):
        """Perform lr scheduler step, if present."""
        # This should be incorporated in the optim step:
        # https://deepspeed.readthedocs.io/en/latest/schedulers.html
        # scheduler is updated automatically at each training step

    @abc.abstractmethod
    def save_checkpoint(self):
        """Save checkpoint to persistent storage."""


class DDPModelEngine(ModelEngine):
    """Model engine for torch DDP distributed strategy."""

    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Performs the forward operation."""
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                return self.model(*args, **kwds)
        else:
            return self.model(*args, **kwds)

    def zero_grad(self):
        """Set gradients to zero for the optimizer."""
        self.optimizer.zero_grad()

    def backward(self, loss_fn: Callable, *loss_args) -> torch.Tensor:
        """Perform backward pass and return the loss.

        Args:
            loss_fn (Callable): computes the loss.
            *loss_args: are the arguments to be passed to ``loss_fn``.

        Returns:
            torch.Tensor: computed loss.
        """
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = loss_fn(*loss_args)

            # Scales loss.  Calls backward() on scaled loss to create scaled
            # gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for
            # corresponding forward ops.
            loss = self.grad_scaler.scale(loss)
        else:
            loss = loss_fn(*loss_args)
        loss.backward()
        return loss

    def optimizer_step(self):
        """Perform optimizer step."""
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
            # scaler.step() first unscales the gradients of the optimizer's
            # assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step()
            # is then called,
            # otherwise, optimizer.step() is skipped.
            self.grad_scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def lr_scheduler_step(self):
        """Perform lr scheduler step, if present."""
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def save_checkpoint(self):
        """Save checkpoint to persistent storage."""
        raise NotImplementedError


class DSModelEngine(ModelEngine):
    """Model engine for DeeSpeed distributed strategy."""

    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Performs the forward operation."""
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                return self.model(*args, **kwds)
        else:
            return self.model(*args, **kwds)

    def zero_grad(self):
        """Set gradients to zero for the optimizer."""
        self.optimizer.zero_grad()

    def backward(self, loss_fn: Callable, *loss_args) -> torch.Tensor:
        """Perform backward pass and return the loss.

        Args:
            loss_fn (Callable): computes the loss.
            *loss_args: are the arguments to be passed to ``loss_fn``.

        Returns:
            torch.Tensor: computed loss.
        """
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = loss_fn(*loss_args)

            # Scales loss.  Calls backward() on scaled loss to create scaled
            # gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for
            # corresponding forward ops.
            loss = self.grad_scaler.scale(loss)
        else:
            loss = loss_fn(*loss_args)
        loss.backward()
        return loss

    def optimizer_step(self):
        """Perform optimizer step."""
        if self.mixed_precision:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
            # scaler.step() first unscales the gradients of the optimizer's
            # assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step()
            # is then called,
            # otherwise, optimizer.step() is skipped.
            self.grad_scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def lr_scheduler_step(self):
        """Perform lr scheduler step, if present."""
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def save_checkpoint(self):
        """Save checkpoint to persistent storage."""
        raise NotImplementedError

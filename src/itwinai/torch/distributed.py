import abc
from typing import Any, Union, List, Dict, Optional, Callable, Tuple
from pathlib import Path
import json

from pydantic import BaseModel

import deepspeed
import torch
import torch.distributed as dist
import horovod.torch as hvd
import torch.nn as nn
# from torch.nn.modules import Module
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.cuda import amp
from torch import autocast

from ..distributed import DistributedStrategy


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
    """Wrapper around distributed model"""

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


class TorchDistributedStrategy(DistributedStrategy):
    """Abstract class to define the distributed backend methods for
    PyTorch models.
    """
    @abc.abstractmethod
    def init(self) -> None:
        """Initializes the chosen distributed backend"""

    # @abc.abstractmethod
    # def distributed_engine(
    #     self, model: nn.Module, optimizer: Optimizer,
    #     lr_scheduler: Optional[LRScheduler] = None
    # ) -> ModelEngine:
    #     """Build a distributed model engine."""

    @abc.abstractmethod
    def distributed(
        self, model: nn.Module, optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler] = None
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""

    @abc.abstractmethod
    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """

    @abc.abstractmethod
    def dist_lwsize(self) -> int:
        """Returns the number of local workers available on a node
        (local world size).
        Usually it is equal to the number of available GPUs.

        Returns:
            int: local world size.
        """

    @abc.abstractmethod
    def dist_grank(self) -> int:
        """Returns the global rank of the current process.
        Rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """

    @abc.abstractmethod
    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """

    def is_main_worker(self) -> bool:
        """Checks if local worker has global rank equal to zero.

        Returns:
            bool: True if main worker.
        """
        return self.dist_grank() == 0

    def dist_device(self) -> str:
        """Device used by local worker.

        Returns:
            str: torch device in the form 'cuda:N'.
        """
        return f"cuda:{self.dist_lrank()}"

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Cleans up resources allocated by distributed strategy."""

    @abc.abstractmethod
    def par_allgather_obj(self, obj: Any) -> List[Any]:
        """Gathers any object from the whole group in a list (to all workers).

        Args:
            obj (Any): object to gather from all workers.

        Returns:
            List[Any]: list of objects gathered from all workers.
        """


class DDPDistributedStrategy(TorchDistributedStrategy):
    """PyTorch DDP distributed strategy class.

    Args:
        backend (str): Name of the communication backend to employ.
    """

    backend: str
    model: DDPModelEngine

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            dist.init_process_group(backend=self.backend)
        else:
            print("WARNING: trying to run distributed on insufficient"
                  " resources. Skipping distributed process group setup.")

    # def distributed_engine(
    #     self, model: nn.Module, optimizer: Optimizer,
    #     lr_scheduler: Optional[LRScheduler] = None,
    #     mixed_precision: bool = False
    # ) -> ModelEngine:
    #     """Build a distributed model engine."""
    #     if torch.cuda.is_available():
    #         # device = self.dist_lrank()
    #         model = model.to(self.dist_device())
    #         dist_model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[self.dist_device()],
    #             output_device=self.dist_device()
    #         )
    #     else:
    #         dist_model = model

    #     model_engine = DDPModelEngine(
    #         dist_model, optimizer, lr_scheduler,
    #         mixed_precision=mixed_precision
    #     )

    #     return model_engine

    def distributed(
        self, model: nn.Module, optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler] = None,
        **kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""
        if torch.cuda.is_available():
            # device = self.dist_lrank()
            model = model.to(self.dist_device())
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist_device()],
                output_device=self.dist_device()
            )
        else:
            dist_model = model

        return dist_model, optimizer, lr_scheduler

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()

    def par_allgather_obj(self, obj: Any) -> List[Any]:
        """Gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class DSDistributedStrategy(TorchDistributedStrategy):
    """DeepSpeed distributed strategy class.

    Args:
        backend (str): Name of the communication backend to employ.
        config (Union[dict, Path, str]): DeepSpeed config. Either a
        dictionary or a path to a JSON file.
    """

    config: Dict = None
    backend: str

    def __init__(
        self,
        backend: str,
        config: Union[Dict, Path, str]
    ) -> None:
        super().__init__()
        self.backend = backend
        self._load_config(config)

    def _load_config(self, ds_config):
        if isinstance(ds_config, (str, Path)):
            with open(ds_config) as fp:
                self.config = json.load(fp)
        elif isinstance(ds_config, dict):
            self.config = ds_config
        else:
            raise ValueError("ds_config is not a dictionary not a path.")

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.
        """
        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        deepspeed.init_distributed(dist_backend=self.backend)

    def distributed(
        self, model: nn.Module, optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        model_parameters: Optional[Any] = None, **kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""
        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        # To prioritize optim in the config, you need to pass optim=None
        distrib_model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
            config=self.config
        )
        return distrib_model, optimizer, lr_scheduler

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        deepspeed.sys.exit()

    def par_allgather_obj(self, obj: Any) -> list[Any]:
        """Gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class HVDDistributedStrategy(TorchDistributedStrategy):
    """Horovod distributed strategy class."""

    def init(self) -> None:
        """Initializes the Horovod distributed backend."""
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    def distributed(
        self, model: nn.Module, optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        **kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""

        model.to(self.dist_device())
        self._broadcast_params(model, optimizer)

        # TODO: here you may need to scale the lr

        distOptimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            op=hvd.Average
        )
        return model, distOptimizer, lr_scheduler

    def _broadcast_params(
            self, model: nn.Module, optimizer: optim.Optimizer
    ) -> None:
        """Broadcasts variables from root rank to all other processes.

        Args:
            model (nn.Module): ML model that is to be broadcasted
            across processes.
            optimizer (optim.Optimizer): Optimizer that is to be broadcasted
            across processes.
        """
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=-0)

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return hvd.size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return hvd.local_size()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return hvd.rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return hvd.local_rank()

    def clean_up(self) -> None:
        """Shuts Horovod down."""
        hvd.shutdown()

    def par_allgather_obj(self, obj: Any) -> list[Any]:
        """Gathers scalar objects across all workers to a
        list with size(#worker), uses horovod communicator

        Args:
            obj (Any): object in a worker.

        Returns:
            list: gathered list with size(#worker).
        """
        return hvd.allgather_object(obj)

###################################################################


class TorchDistributedStrategy_old(DistributedStrategy):
    """Abstract class to define the distributed backend methods for
    PyTorch models.
    """
    @abc.abstractmethod
    def init_backend(self) -> None:
        """Initializes the chosen distributed backend"""

    @abc.abstractmethod
    def distribute_model(self, model: Any) -> Any:
        """Distributes a machine learning model.

        Args:
            model (Any): a generic ML model to be distributed.

        Returns:
            Any: distributed model instance.
        """

    @abc.abstractmethod
    def broadcast_params(self, model: Any, optimizer: Any) -> None:
        """Broadcasts variables from root rank to all other processes/

        Args:
            model (Any): distributed model.
            optimizer (Any): optimizer.
        """

    @abc.abstractmethod
    def distribute_optimizer(self, optimizer: Any, model: Any) -> Any:
        """Distribute optimizer.

        Args:
            optimizer (Any): optimizer.
            model (Any): distributed model.

        Returns:
            Any: distributed optimizer.
        """

    @abc.abstractmethod
    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """

    @abc.abstractmethod
    def dist_lwsize(self) -> int:
        """Returns the number of local workers available on a node
        (local world size).
        Usually it is equal to the number of available GPUs.

        Returns:
            int: local world size.
        """

    @abc.abstractmethod
    def dist_grank(self) -> int:
        """Returns the global rank of the current process.
        Rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """

    @abc.abstractmethod
    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """

    def is_main_worker(self) -> bool:
        """Checks if local worker has global rank equal to zero.

        Returns:
            bool: True if main worker.
        """
        return self.dist_grank() == 0

    def dist_device(self) -> str:
        """Device used by local worker.

        Returns:
            str: torch device in the form 'cuda:N'.
        """
        return f"cuda:{self.dist_lrank()}"

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Cleans up resources allocated by distributed strategy."""

    @abc.abstractmethod
    def par_allgather_obj(self, obj: Any) -> List[Any]:
        """Gathers any object from the whole group in a list (to all workers).

        Args:
            obj (Any): object to gather from all workers.

        Returns:
            List[Any]: list of objects gathered from all workers.
        """


class DDPDistributedStrategy_old(TorchDistributedStrategy_old):
    """PyTorch DDP distributed strategy class.

    Args:
        backend (str): Name of the communication backend to employ.
    """

    backend: str

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend

    def init_backend(self) -> None:
        """Initializes the distributed process group and the distributed
        package.
        """
        if torch.cuda.is_available():
            dist.init_process_group(backend=self.backend)

    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Achieves data parallelism by synchronizing the gradients
        across each model replica located in each available
        computing device.

        Args:
            model (nn.Module): ML model to be distributed.

        Returns:
            nn.Module: Distributed model replicas across all devices.
            that are to be synchronized.
        """
        if torch.cuda.is_available():
            # device = self.dist_lrank()
            model = model.to(self.dist_device())
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist_device()],
                output_device=self.dist_device()
            )
        else:
            dist_model = model

        return dist_model

    def broadcast_params(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer
    ) -> None:
        """Do nothing. Only applicable for Horovod.

        Args:
            model (nn.Module): ML model
            optimizer (optim.Optimizer): Optimizer
        """
        pass

    def distribute_optimizer(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module = None
    ) -> optim.Optimizer:
        """Returns the optimizer from argument.

        Args:
            optimizer (optim.Optimizer): optimizer.
            model (nn.Module): ML model. Unused here.

        Returns:
            optim.Optimizer: Distributed optimizer.
        """
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()

    def par_allgather_obj(self, obj: Any) -> List[Any]:
        """Gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class DSDistributedStrategy_old(TorchDistributedStrategy_old):
    """DeepSpeed distributed strategy class.

    Args:
        backend (str): Name of the communication backend to employ.
        config (Union[dict, Path, str]): DeepSpeed config. Either a
        dictionary or a path to a JSON file.
    """

    config: Dict = None
    backend: str

    def __init__(
        self,
        backend: str,
        config: Union[Dict, Path, str]
    ) -> None:
        super().__init__()
        self.backend = backend
        self._load_config(config)

    def _load_config(self, ds_config):
        if isinstance(ds_config, (str, Path)):
            with open(ds_config) as fp:
                self.config = json.load(fp)
        elif isinstance(ds_config, dict):
            self.config = ds_config
        else:
            raise ValueError("ds_config is not a dictionary not a path.")

    def init_backend(self) -> None:
        """Initializes the distributed process group and the distributed
        package.
        """
        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        deepspeed.init_distributed(dist_backend=self.backend)

    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Achieves data parallelism by synchronizing the gradients
        across each model replica located in each available
        computing device.

        Args:
            model (nn.Module): ML model to be distributed.

        Returns:
            nn.Module: Distributed model replicas across all devices
            that are to be synchronized.
        """
        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        distrib_model, __, __, __ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=True,
            config=self.config
        )
        return distrib_model

    def broadcast_params(
            self, model: nn.Module, optimizer: optim.Optimizer
    ) -> None:
        """Only applicable for Horovod. Does nothing.

        Args:
            model (nn.Module): ML model.
            optimizer (optim.Optimizer): optimizer.
        """
        pass

    def distribute_optimizer(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module = None
    ) -> optim.Optimizer:
        """Returns the optimizer from argument.

        Args:
            optimizer (optim.Optimizer): torch optimizer.
            model (nn.Module): torch neural network.

        Returns:
            optim.Optimizer: distributed optimizer.
        """
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        deepspeed.sys.exit()

    def par_allgather_obj(self, obj: Any) -> list[Any]:
        """Gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class HVDDistributedStrategy_old(TorchDistributedStrategy_old):
    """Horovod distributed strategy class."""

    def init_backend(self) -> None:
        """Initializes the Horovod distributed backend."""
        hvd.init()

    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Only applicable for DDP and DeepSpeed.
        For Horovod, returns the same model passed as argument.

        Args:
            model (nn.Module): ML model to be distributed.

        Returns:
            nn.Module: ML model passed in the argument.
        """
        return model

    def broadcast_params(
            self, model: nn.Module, optimizer: optim.Optimizer
    ) -> None:
        """Broadcasts variables from root rank to all other processes.

        Args:
            model (nn.Module): ML model that is to be broadcasted
            across processes.
            optimizer (optim.Optimizer): Optimizer that is to be broadcasted
            across processes.
        """
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=-0)

    def distribute_optimizer(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module
    ) -> optim.Optimizer:
        """Constructs a DistributedOptimizer, for computing single-process
        gradient values and applying gradient updates after the gradient values
        have been combined across all the Horovod ranks.

        Args:
            optimizer (optim.Optimizer): Optimizer to be distributed.
            model (nn.Module): ML model to be trained.

        Returns:
            optim.Optimizer: Distributed optimizer across all ranks.
        """
        distOptimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            op=hvd.Average
        )
        return distOptimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return hvd.size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return hvd.local_size()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return hvd.rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return hvd.local_rank()

    def clean_up(self) -> None:
        """Shuts Horovod down."""
        hvd.shutdown()

    def par_allgather_obj(self, obj: Any) -> list[Any]:
        """Gathers scalar objects across all workers to a
        list with size(#worker), uses horovod communicator

        Args:
            obj (Any): object in a worker.

        Returns:
            list: gathered list with size(#worker).
        """
        return hvd.allgather_object(obj)

import abc
from typing import Any, Union, List, Dict, Optional, Tuple
from pathlib import Path
import json
import os

import deepspeed
import torch
import torch.distributed as dist
import horovod.torch as hvd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

from ..distributed import DistributedStrategy


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
        # https://github.com/Lightning-AI/pytorch-lightning/issues/13567
        ompi_lrank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get(
            'LOCAL_RANK', ompi_lrank)

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

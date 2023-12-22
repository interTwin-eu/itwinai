import abc
from typing import Any, Union, List

import deepspeed
import torch
import torch.distributed as dist
import horovod.torch as hvd
import torch.nn as nn
import torch.optim as optim

from ..distributed import DistributedStrategy


class TorchDistributedStrategy(DistributedStrategy):
    """Abstract class to define the distributed backend methods for
    PyTorch models.
    """
    @abc.abstractmethod
    def init_backend(self, *args, **kwargs) -> None:
        """Initializes the chosen distributed backend"""

    @abc.abstractmethod
    def distribute_model(self, model: Any, device: Union[int, str]) -> Any:
        """Distributes a machine learning model.

        Args:
            model (Any): a generic ML model to be distributed.
            device (Union[int, str]): device on which the model is run.

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
        """Returns the total number of processes (global word size).

        Returns:
            int: global word size.
        """

    @abc.abstractmethod
    def dist_lwsize(self) -> int:
        """Returns the number of local workers available on a node
        (local word size). Usually it is equal to the number of available GPUs.

        Returns:
            int: local word size.
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
    """PyTorch DDP distributed training class

    Args:
        TorchDistributedStrategy (Any): Parent class
    """

    def init_backend(self, backend: str, *args, **kwargs) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Args:
            backend (str): Name of the communication backend to employ
        """
        if torch.cuda.is_available():
            dist.init_process_group(backend=backend)

    def distribute_model(self, model, device) -> nn.Module:
        """Achieves data parallelism by synchronizing the gradients 
        across each model replica located in each available 
        computing device.

        Args:
            model (nn.Module): ML model to be distributed
            device (Union[int, str]): Compute device to be used

        Returns:
            nn.Module: Distributed model replicas across all devices 
            that are to be synchronized
        """
        if torch.cuda.is_available():
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device],
                output_device=device
            )
        else:
            dist_model = model

        return dist_model

    def broadcast_params(
        self,
        model,
        optimizer
    ) -> None:
        """Only applicable for Horovod, else pass

        Args:
            model (Any): ML model
            optimizer (Any): Optimizer
        """
        pass

    def distribute_optimizer(
        self,
        optimizer,
        model
    ) -> optim.Optimizer:
        """Only applicable for Horovod, else returns the
        optimizer from argument

        Args:
            optimizer (Any): Optimizer
            model (Any): ML model

        Returns:
            optim.Optimizer: Distributed optimizer
        """
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size)

        Returns:
            int: global world size
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available

        Returns:
            int: local world size
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size

        Returns:
            int: global rank
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process

        Returns:
            int: local rank
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group
        """
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()

    def par_allgather_obj(self, obj) -> List[Any]:
        """Gathers any object from the whole group 
        in a list (to all workers)

        Args:
            obj (Any): Object to gather from all workers

        Returns:
            List[Any]: List of gathered objects
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class DSDistributedStrategy(TorchDistributedStrategy):
    """DeepSpeed distributed training class

    Args:
        TorchDistributedStrategy (Any): Parent class
    """

    def init_backend(self, backend: str, *args, **kwargs) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Args:
            backend (str): Name of the communication backend to employ
        """
        deepspeed.init_distributed(dist_backend=backend)

    def distribute_model(self, model, device) -> nn.Module:
        """Achieves data parallelism by synchronizing the gradients 
        across each model replica located in each available 
        computing device.

        Args:
            model (nn.Module): ML model to be distributed
            device (Union[int, str]): Compute device to be used

        Returns:
            nn.Module: Distributed model replicas across all devices 
            that are to be synchronized
        """
        distrib_model, __, __, __ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=True
        )
        return distrib_model

    def broadcast_params(self, distrib_model, optimizer) -> None:
        """Only applicable for Horovod, else pass

        Args:
            model (Any): ML model
            optimizer (Any): Optimizer
        """
        pass

    def distribute_optimizer(
        self,
        optimizer,
        distrib_model
    ) -> optim.Optimizer:
        """Only applicable for Horovod, else returns the
        optimizer from argument

        Args:
            optimizer (Any): Optimizer
            model (Any): ML model

        Returns:
            optim.Optimizer: Distributed optimizer
        """
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size)

        Returns:
            int: global world size
        """
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available

        Returns:
            int: local world size
        """
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size

        Returns:
            int: global rank
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process

        Returns:
            int: local rank
        """
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group
        """
        deepspeed.sys.exit()

    def par_allgather_obj(self, obj) -> list:
        """Gathers any object from the whole group 
        in a list (to all workers)

        Args:
            obj (Any): Object to gather from all workers

        Returns:
            List[Any]: List of gathered objects
        """
        res = [None] * self.dist_gwsize()
        dist.all_gather_object(res, obj)
        return res


class HVDDistributedStrategy(TorchDistributedStrategy):
    """Horovod distributed training class

    Args:
        TorchDistributedStrategy (Any): Parent class
    """

    def init_backend(self, *args, **kwargs) -> None:
        """Initializes the Horovod distributed backend
        """
        hvd.init()

    def distribute_model(self, model, device) -> nn.Module:
        """Only applicable for DDP and DeepSpeed.
        For Horovod, returns the same model passed as argument

        Args:
            model (nn.Module): ML model to be distributed
            device (Union[int, str]): Compute device to be used

        Returns:
            nn.Module: ML model passed in the argument
        """
        distrib_model = model
        return distrib_model

    def broadcast_params(self, distrib_model, optimizer) -> None:
        """Broadcasts variables from root rank to all other processes

        Args:
            distrib_model (nn.Module): ML model that is to be broadcasted 
            across processes
            optimizer (optim.Optimizer): Optimizer that is to be broadcasted 
            across processes  
        """
        hvd.broadcast_parameters(distrib_model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=-0)

    def distribute_optimizer(
        self,
        optimizer,
        distrib_model
    ) -> optim.Optimizer:
        """Constructs a DistributedOptimizer, for computing single-process
        gradient values and applying gradient updates after the gradient values
        have been combined across all the Horovod ranks.

        Args:
            optimizer (optim.Optimizer): Optimizer to be distributed
            distrib_model (nn.Module): ML model to be trained

        Returns:
            optim.Optimizer: Distributed optimizer across all ranks 
        """
        distOptimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=distrib_model.named_parameters(),
            op=hvd.Average
        )
        return distOptimizer

    def dist_gwsize(self) -> int:
        """Returns the total number of processes (global world size)

        Returns:
            int: global world size
        """
        return hvd.size()

    def dist_lwsize(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available

        Returns:
            int: local world size
        """
        return hvd.local_size()

    def dist_grank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size

        Returns:
            int: global rank
        """
        return hvd.rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process

        Returns:
            int: local rank
        """
        return hvd.local_rank()

    def clean_up(self) -> None:
        """Shuts Horovod down
        """
        hvd.shutdown()

    def par_allgather_obj(self, obj, gwsize) -> list:
        """Gathers scalar objects across all workers to a 
        list with size(#worker), uses horovod communicator

        Args:
            obj (Any): object in a worker
            gwsize (int): global world size

        Returns:
            list: gathered list with size(#worker)
        """
        return hvd.allgather_object(obj)

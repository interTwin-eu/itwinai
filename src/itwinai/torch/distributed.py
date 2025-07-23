# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Rakesh Sarma <r.sarma@fz-juelich.de> - Juelich
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import abc
import functools
import logging
import os
from typing import Any, Callable, Iterable, List, Literal, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from ..distributed import detect_distributed_environment, ray_cluster_is_running
from .type import DistributedStrategyError, UninitializedStrategyError

py_logger = logging.getLogger(__name__)


def distributed_resources_available() -> bool:
    """Check if the current execution environment
    has (enough) GPUs available to allow for distributed ML.

    Returns:
        bool: env can support distributed ML.
    """
    if int(os.environ.get("ITWINAI_FORCE_DIST", "0")):
        return True
    cluster = detect_distributed_environment()
    return cluster.global_world_size > 1


def check_initialized(method: Callable) -> Callable:
    """Decorator for strategy methods to check whether the strategy
    was correctly initialized before calling the method."""

    @functools.wraps(method)
    def wrapper(self: "TorchDistributedStrategy", *args, **kwargs):
        if not self.is_initialized:
            raise UninitializedStrategyError(
                f"{self.__class__.__name__} has not been initialized. Use the init method."
            )
        return method(self, *args, **kwargs)

    return wrapper


def initialize_ray() -> None:
    """This method is used by the RayDDPStrategy and RayDeepSpeedStrategy to initialize
    the Ray backend if it is not already initialized. This is meant to be called before
    submitting a function to Ray (as a trial in tuning, or as a worker in distributed ML).

    Raises:
        RuntimeError: when no Ray cluster is detected.
        EnvironmentError: If required environment variables `HEAD_NODE_PORT` or
            `HEAD_NODE_IP` are not set.
            These should be set from the slurm script where the ray cluster is launched.
    """
    import ray
    from ray.runtime_env import RuntimeEnv

    if not ray_cluster_is_running():
        raise RuntimeError(
            "You are trying to initialize Ray, but the cluster seems not to be running"
        )

    if ray.is_initialized():
        return

    mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    if not mlflow_username:
        py_logger.warning("MLFLOW_TRACKING_USERNAME env variable is not set.")
    if not mlflow_password:
        py_logger.warning("MLFLOW_TRACKING_PASSWORD env variable is not set.")

    # Set mlflow credentials to be accessible for all the workers
    runtime_env = RuntimeEnv(
        env_vars={
            "MLFLOW_TRACKING_USERNAME": mlflow_username,
            "MLFLOW_TRACKING_PASSWORD": mlflow_password,
        }
    )
    ray.init(address="auto", runtime_env=runtime_env)
    py_logger.info(f"Nodes in the cluster: {ray.nodes()}")
    py_logger.info(f"Available cluster resources: {ray.available_resources()}")


class TorchDistributedStrategy(abc.ABC):
    """Abstract class to define the distributed backend methods for
    PyTorch models.
    """

    #: Allows to discriminate distributed strategies from non-distributed.
    #: Defaults to True.
    is_distributed: bool = True
    #: Set to True when the current strategy is initialized.
    #: Defaults to False.
    is_initialized: bool = False

    # Provides the name of the strategy for logging purposes etc.
    name: str

    @property
    @check_initialized
    def is_main_worker(self) -> bool:
        """Checks if local worker has global rank equal to zero.

        Returns:
            bool: True if main worker.
        """
        return self.global_rank() == 0

    @abc.abstractmethod
    def init(self) -> None:
        """Initializes the chosen distributed backend"""

    @abc.abstractmethod
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None = None,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        """Setup model, optimizer and scheduler for distributed."""

    @abc.abstractmethod
    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """

    @abc.abstractmethod
    def local_world_size(self) -> int:
        """Returns the number of local workers available on a node
        (local world size).
        Usually it is equal to the number of available GPUs.

        Returns:
            int: local world size.
        """

    @abc.abstractmethod
    def global_rank(self) -> int:
        """Returns the global rank of the current process.
        Rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """

    @abc.abstractmethod
    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """

    @abc.abstractmethod
    def barrier(self) -> None:
        """Forces all the workers to wait for each other."""

    @check_initialized
    def device(self) -> str:
        """Device used by local worker.

        Returns:
            str: torch device in the form 'device:N' (e.g., 'cuda:0', 'cpu').
        """
        if torch.cuda.is_available():
            return f"cuda:{self.local_rank()}"
        return "cpu"

    def set_device(self):
        """Set local device."""
        if torch.cuda.is_available():
            torch.cuda.device(self.local_rank())
            # Needed by torch.distributed.gather_object
            torch.cuda.set_device(self.local_rank())

    @check_initialized
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """Create a distributed DataLoader by using ``DistributedSampler`` as
        random sampler.

        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: ``False``).
            sampler (Sampler or Iterable, optional): defines the strategy to
                draw
                samples from the dataset. Can be any ``Iterable`` with
                ``__len__``
                implemented. If specified, :attr:`shuffle` must not be
                specified.
            batch_sampler (Sampler or Iterable, optional): like
                :attr:`sampler`, but
                returns a batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
                and :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main
                process. (default: ``0``)
            collate_fn (Callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from
                a map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will
                copy Tensors
                into device/CUDA pinned memory before returning them.  If your
                data elements
                are a custom type, or your :attr:`collate_fn` returns a batch
                that is a custom type,
                see the example below.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch,
                if the dataset size is not divisible by the batch size.
                If ``False`` and
                the size of dataset is not divisible by the batch size, then
                the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn (Callable, optional): If not ``None``,
                this will be called on each
                worker subprocess with the worker id (an int in
                ``[0, num_workers - 1]``) as
                input, after seeding and before data loading.
                (default: ``None``)
            multiprocessing_context (str or
                multiprocessing.context.BaseContext, optional): If
                ``None``, the default `multiprocessing context`_ of
                your operating system will
                be used. (default: ``None``)
            generator (torch.Generator, optional): If not ``None``,
                this RNG will be used
                by RandomSampler to generate random indexes and
                multiprocessing to generate
                ``base_seed`` for workers. (default: ``None``)
            prefetch_factor (int, optional, keyword-only arg): Number of
                batches loaded
                in advance by each worker. ``2`` means there will be a total of
                2 * num_workers batches prefetched across all workers.
                (default value depends
                on the set value for num_workers. If value of num_workers=0
                default is ``None``.
                Otherwise, if value of ``num_workers > 0`` default is ``2``).
            persistent_workers (bool, optional): If ``True``, the data loader
                will not shut down
                the worker processes after a dataset has been consumed once.
                This allows to
                maintain the workers `Dataset` instances alive.
                (default: ``False``)
            pin_memory_device (str, optional): the device to
                :attr:`pin_memory` to if ``pin_memory`` is ``True``.

        Raises:
            UninitializedStrategyError: when this method is called for a
                strategy which had not been initialized.
            RuntimeError: when a user-provided sampler, if given, is not of
                type ``DistributedSampler``.

        .. warning:: If the ``spawn`` start method is used,
                    :attr:`worker_init_fn`
                    cannot be an unpicklable object, e.g., a lambda function.
                    See `Multiprocessing best practices`_ on more
                    details related to multiprocessing in PyTorch.


        .. warning:: ``len(dataloader)`` heuristic is based on the length of
                    the sampler used.
                    When :attr:`dataset` is an
                    :class:`~torch.utils.data.IterableDataset`,
                    it instead returns an estimate based on
                    ``len(dataset) / batch_size``, with proper
                    rounding depending on :attr:`drop_last`, regardless
                    of multi-process loading
                    configurations. This represents the best guess PyTorch
                    can make because PyTorch
                    trusts user :attr:`dataset` code in correctly handling
                    multi-process
                    loading to avoid duplicate data.

                    However, if sharding results in multiple workers having
                    incomplete last batches,
                    this estimate can still be inaccurate, because (1) an
                    otherwise complete batch can
                    be broken into multiple ones and (2) more than one batch
                    worth of samples can be
                    dropped when :attr:`drop_last` is set. Unfortunately,
                    PyTorch can not detect such cases in general.

                    See `Dataset Types`_ for more details on these two
                    types of datasets and how
                    :class:`~torch.utils.data.IterableDataset` interacts with
                    `Multi-process data loading`_.


        .. warning:: See `Reproducibility`_, and
                    `My data loader workers return identical random numbers`_,
                    and
                    `Randomness in multi-process data loading`_ notes for
                    random seed related questions.


        .. _multiprocessing context:
            https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        .. _Multiprocessing best practices:
            https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-best-practices
        .. _Reproducibility:
            https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
        .. _My data loader workers return identical random numbers:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
        .. _Randomness in multi-process data loading:
            https://pytorch.org/docs/stable/data.html#data-loading-randomness
        .. _Multi-process data loading:
            https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        .. _Dataset Types:
            https://pytorch.org/docs/stable/data.html#dataset-types
        """

        if batch_sampler is not None:
            py_logger.warning("WARNING: batch_sampler is ignored by TorchDistributedStrategy")

        if self.is_distributed:
            if sampler is None:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.global_world_size(),
                    rank=self.global_rank(),
                    shuffle=shuffle,
                )
            elif not isinstance(sampler, DistributedSampler):
                raise RuntimeError("User-provided sampler must implement DistributedSampler.")
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        # shuffle and batch_sampler must be unset
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Cleans up resources allocated by distributed strategy."""

    @abc.abstractmethod
    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gathers any object from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.

        Returns:
            List[Any]: list of objects gathered from all workers.
        """

    @abc.abstractmethod
    def gather_obj(self, obj: Any, dst_rank: int = 0) -> List[Any]:
        """Gathers any object from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            List[Any]: list of objects gathered from all workers.
        """

    @abc.abstractmethod
    def gather(self, tensor: torch.Tensor, dst_rank: int = 0) -> List | None:
        """Gathers any object from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            Optional[List[Any]]: list of objects gathered from all workers if main
                worker, otherwise return None.
        """

    @abc.abstractmethod
    def broadcast_obj(self, obj: Any, src_rank: int) -> Any:
        """Broadcasts an object to all workers.

        Args:
            obj (Any): object to broadcast to all workers.
            src_rank (int): the rank that broadcasted

        Returns:
            Any: broadcasted object.
        """


class TorchDDPStrategy(TorchDistributedStrategy):
    """PyTorch ``DistributedDataParallel`` distributed strategy class.

    Args:
        backend (Literal['nccl', 'gloo', 'mpi']): Name of the
            distributed communication backend to employ.
    """

    #: Torch distributed communication backend.
    backend: Literal["nccl", "gloo", "mpi"]

    def __init__(self, backend: Literal["nccl", "gloo", "mpi"]) -> None:
        super().__init__()
        self.backend = backend
        self.name = "torch-ddp"

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                which is already initialized.
        """
        if not distributed_resources_available():
            raise RuntimeError("Trying to run distributed on insufficient resources.")
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        dist.init_process_group(backend=self.backend)
        self.is_initialized = True

        self.set_device()

    @check_initialized
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None = None,
        find_unused_parameters: bool = False,
        **kwargs,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        """Setup model, optimizer and scheduler for distributed."""

        if torch.cuda.is_available():
            # If GPUs are available
            model = model.to(self.device())
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device()],
                output_device=self.device(),
                find_unused_parameters=find_unused_parameters,
            )
        elif distributed_resources_available():
            # If GPUs are not available, but running distributed ML on CPUs
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            dist_model = model

        return dist_model, optimizer, lr_scheduler

    @check_initialized
    def barrier(self) -> None:
        """Forces all the workers to wait for each other."""
        return dist.barrier()

    @check_initialized
    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    @check_initialized
    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.

        Raises:
            RuntimeError: when the local world size cannot be retrieved.
        """
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        if "LOCAL_WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "Could not retrieve local world size as CUDA is unavailable and there is "
                "no 'LOCAL_WORLD_SIZE' environment variable."
            )
        return int(os.environ["LOCAL_WORLD_SIZE"])

    @check_initialized
    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    @check_initialized
    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % self.local_world_size()

    @check_initialized
    def clean_up(self) -> None:
        """Destroys the current process group."""
        if distributed_resources_available():
            dist.barrier()
            dist.destroy_process_group()

    @check_initialized
    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        res = [None] * self.global_world_size()
        dist.all_gather_object(res, obj)
        return res

    @check_initialized
    def gather_obj(self, obj: Any, dst_rank: int = 0) -> List | None:
        """Gathers any object from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            List | None: list of objects gathered from all workers
            or ``None`` on non-destination ranks.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        if self.global_rank() == dst_rank:
            res = [None] * self.global_world_size()
            dist.gather_object(obj, res, dst=dst_rank)
            return res

        dist.gather_object(obj, dst=dst_rank)

    @check_initialized
    def gather(self, tensor: torch.Tensor, dst_rank: int = 0) -> List | None:
        # https://pytorch.org/docs/stable/distributed.html#collective-functions

        # Ensure that the tensor is on the correct device (CUDA)
        tensor = tensor.to(self.device())
        if self.global_rank() != dst_rank:
            dist.gather(tensor, dst=dst_rank)
            return

        res = [
            torch.zeros_like(tensor, device=self.device())
            for _ in range(self.global_world_size())
        ]

        dist.gather(tensor, gather_list=res, dst=dst_rank)

        # Moving everything to the CPU before returning
        return [val.cpu() for val in res]

    @check_initialized
    def broadcast_obj(self, obj: Any, src_rank: int) -> Any:
        """Broadcasts an object to all workers. (object must be picklable)

        Args:
            obj (Any): object to broadcast to all workers.
            src_rank (int): the rank that broadcasted

        Returns:
            Any: broadcasted object.
        """
        obj_list = [obj]
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        dist.broadcast_object_list(obj_list, src=src_rank)
        return obj_list[0]


class DeepSpeedStrategy(TorchDistributedStrategy):
    """DeepSpeed distributed strategy class.

    Args:
        backend (Literal['nccl', 'gloo', 'mpi']): Name of the
            distributed communication backend to employ.
        config (Union[dict, Path, str]): DeepSpeed config. Either a
            dictionary or a path to a JSON file.
    """

    #: Torch distributed communication backend.
    backend: Literal["nccl", "gloo", "mpi"]

    def __init__(self, backend: Literal["nccl", "gloo", "mpi"]) -> None:
        super().__init__()
        self.backend = backend
        self.name = "deepspeed"

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        import deepspeed

        # Removing the .put() method of the cache manager
        # This is the same bug that was earlier removed in the generic_torch.sh script,
        # using the sed command
        from deepspeed.ops.transformer.inference.triton.matmul_ext import AutotuneCacheManager

        def noop_put(self, table):
            pass

        AutotuneCacheManager.put = noop_put
        py_logger.warning(
            "[WARNING]: Disabling Triton's AutotuneCacheManager's `put()` method to fix "
            "bug with temporary files. This might be fixed in the future by DeepSpeed,"
            "in which case our fix should be removed."
        )

        self.deepspeed = deepspeed
        if not distributed_resources_available():
            raise RuntimeError("Trying to run distributed on insufficient resources.")

        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")

        # https://github.com/Lightning-AI/pytorch-lightning/issues/13567
        # This block of code should be removed as some point
        if os.environ.get("LOCAL_RANK"):
            os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK")

        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        self.deepspeed.init_distributed(dist_backend=self.backend)
        self.is_initialized = True

        self.set_device()

    @check_initialized
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        model_parameters: Any | None = None,
        **init_kwargs,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        """Setup model, optimizer and scheduler for distributed."""
        py_logger.debug(f"Distributing the model using device: {self.device()}")
        # model = model.to(self.device())

        distrib_model, optimizer, _, lr_scheduler = self.deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
            **init_kwargs,
        )
        return distrib_model, optimizer, lr_scheduler

    @check_initialized
    def barrier(self) -> None:
        """Forces all the workers to wait for each other."""
        return dist.barrier()

    @check_initialized
    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return dist.get_world_size()

    @check_initialized
    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.

        Raises:
            RuntimeError: when the local world size cannot be retrieved.
        """
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        if "LOCAL_WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "Could not retrieve local world size as CUDA is unavailable and there is "
                "no 'LOCAL_WORLD_SIZE' environment variable."
            )
        return int(os.environ["LOCAL_WORLD_SIZE"])

    @check_initialized
    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return dist.get_rank()

    @check_initialized
    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return dist.get_rank() % self.local_world_size()

    @check_initialized
    def clean_up(self) -> None:
        """Destroys the current process group."""
        if distributed_resources_available():
            dist.barrier()
            dist.destroy_process_group()

    @check_initialized
    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        res = [None] * self.global_world_size()
        dist.all_gather_object(res, obj)
        return res

    @check_initialized
    def gather_obj(self, obj: Any, dst_rank: int = 0) -> List[Any] | None:
        """Gathers any object from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            Optional[List[Any]]: list of objects gathered from all workers
            or ``None`` on non-destination ranks.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        if self.global_rank() == dst_rank:
            res = [None] * self.global_world_size()
            dist.gather_object(obj, res, dst=dst_rank)
            return res

        dist.gather_object(obj, dst=dst_rank)

    @check_initialized
    def gather(self, tensor: torch.Tensor, dst_rank: int = 0) -> List[torch.Tensor] | None:
        """Gathers a tensor from the whole group in a list
        (to all workers).

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            Optional[List[torch.Tensor]]: list of tensors gathered from all workers
            or ``None`` on non-destination ranks.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions

        # Ensure that the tensor is on the correct device (CUDA)
        tensor = tensor.to(self.device())
        if self.global_rank() != dst_rank:
            dist.gather(tensor, dst=dst_rank)
            return

        res = [
            torch.zeros_like(tensor, device=self.device())
            for _ in range(self.global_world_size())
        ]

        dist.gather(tensor, gather_list=res, dst=dst_rank)

        # Moving all the tensors to CPU before returning
        return [val.cpu() for val in res]

    @check_initialized
    def broadcast_obj(self, obj: Any, src_rank: int) -> Any:
        """Broadcasts an object to all workers. (object must be picklable)

        Args:
            obj (Any): object to broadcast to all workers.
            src_rank (int): the rank that broadcasted

        Returns:
            Any: broadcasted object.
        """
        obj_list = [obj]
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        dist.broadcast_object_list(obj_list, src=src_rank)
        return obj_list[0]


class HorovodStrategy(TorchDistributedStrategy):
    """Horovod distributed strategy class."""

    def __init__(self):
        super().__init__()
        self.name = "horovod"

    def init(self) -> None:
        """Initializes the Horovod distributed backend.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        if not distributed_resources_available():
            raise RuntimeError("Trying to run distributed on insufficient resources.")
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")

        import horovod.torch as hvd

        self.hvd = hvd

        self.hvd.init()
        self.is_initialized = True

        self.set_device()

    @check_initialized
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        **optim_kwargs,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        """Setup model, optimizer and scheduler for distributed."""

        model.to(self.device())

        # Scale learning rate
        # https://github.com/horovod/horovod/issues/1653#issuecomment-574764452
        lr_scaler = 1
        if optim_kwargs.get("op") == self.hvd.Adasum:
            lr_scaler = self.hvd.local_size()
        elif optim_kwargs.get("op") == self.hvd.Average:
            lr_scaler = self.hvd.size()
        for g in optimizer.param_groups:
            g["lr"] *= lr_scaler

        self._broadcast_params(model, optimizer)

        distOptimizer = self.hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(), **optim_kwargs
        )
        return model, distOptimizer, lr_scheduler

    @check_initialized
    def barrier(self) -> None:
        """Forces all the workers to wait for each other."""
        self.hvd.barrier()

    def _broadcast_params(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Broadcasts variables from root rank to all other processes.

        Args:
            model (nn.Module): ML model that is to be broadcasted
                across processes.
            optimizer (optim.Optimizer): Optimizer that is to be broadcasted
                across processes.
        """
        self.hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        self.hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    @check_initialized
    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return self.hvd.size()

    @check_initialized
    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return self.hvd.local_size()

    @check_initialized
    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return self.hvd.rank()

    @check_initialized
    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return self.hvd.local_rank()

    @check_initialized
    def clean_up(self) -> None:
        """Shuts Horovod down."""
        self.hvd.barrier()
        self.hvd.shutdown()

    @check_initialized
    def allgather_obj(self, obj: Any) -> list[Any]:
        """All-gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        return self.hvd.allgather_object(obj)

    @check_initialized
    def gather_obj(self, obj: Any, dst_rank: int = 0) -> list[Any] | None:
        """Gathers any object from the whole group in a list
        (to all workers). Under the hood it relies on allgather as gather is
        not supported by Horovod.

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            Optional[List[Any]]: list of objects gathered from all workers
            or ``None`` on non-destination ranks.
        """
        result = self.allgather_obj(obj)
        if self.global_rank() == dst_rank:
            # Return only if on rank == dst_rank
            return result

    @check_initialized
    def gather(self, tensor: torch.Tensor, dst_rank: int = 0) -> List[torch.Tensor] | None:
        """Gathers a tensor from the whole group in a list
        (to all workers). Under the hood it relies on allgather as gather is
        not supported by Horovod.

        Args:
            obj (Any): object to gather from all workers.
            dst_rank (int): rank of the worker on which the objects list
                are gathered.

        Returns:
            Optional[List[torch.Tensor]]: list of tensors gathered from all workers
            or ``None`` on non-destination ranks.
        """
        result = self.allgather_obj(tensor)
        if self.global_rank() == dst_rank:
            # Return only if on rank == dst_rank
            # Moving all the tensors to CPU before returning
            return [val.cpu() for val in result]

    @check_initialized
    def broadcast_obj(self, obj: Any, src_rank: int) -> Any:
        """Broadcasts an object to all workers. (object must be picklable)

        Args:
            obj (Any): object to broadcast to all workers.
            src_rank (int): the rank that broadcasted

        Returns:
            Any: broadcasted object.
        """
        if obj is None:
            py_logger.warning(
                "Broadcasting None object in Horovod. This might lead to unexpected behavior"
                " such as deadlocks."
            )
        # https://horovod.readthedocs.io/en/stable/_modules/horovod/torch/functions.html#broadcast_object
        return self.hvd.broadcast_object(obj, root_rank=src_rank)




class NonDistributedStrategy(TorchDistributedStrategy):
    """Dummy class for non-distributed environments."""

    #: This strategy is not distributed.
    #: Defaults to False.
    is_distributed: bool = False

    def __init__(self):
        super().__init__()
        self.name = "non-distributed"

    def init(self) -> None:
        """If CUDA is available set CUDA device, and do nothing more.

        Raises:
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        self.set_device()
        self.is_initialized = True

    @check_initialized
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        **kwargs,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        """Do nothing and return model, optimizer and scheduler."""
        if torch.cuda.is_available():
            model = model.cuda()
        return model, optimizer, lr_scheduler

    @check_initialized
    def barrier(self) -> None:
        """Forces all the workers to wait for each other."""

    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        return 1

    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        return 1

    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        return 0

    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        return 0

    def clean_up(self) -> None:
        """Do nothing."""

    def allgather_obj(self, obj: Any) -> list[Any]:
        """Wraps ``obj`` into a List object.

        Args:
            obj (Any): object in a worker.

        Returns:
            list[Any]: input object wrapped in a list.
        """
        return [obj]

    def gather_obj(self, obj: Any, dst_rank: int = 0) -> list[Any]:
        """Wraps ``obj`` into a List object.

        Args:
            obj (Any): object in a worker.
            dst_rank (int): ignored.

        Returns:
            list[Any]: input object wrapped in a list.
        """
        return [obj]

    def gather(self, tensor: torch.Tensor, dst_rank: int = 0):
        """Wraps ``tensor`` into a List object.

        Args:
            tensor (Any): object in a worker.
            dst_rank (int): ignored.

        Returns:
            list[Any]: input object wrapped in a list.
        """
        return [tensor]

    def broadcast_obj(self, obj: Any, src_rank: int) -> Any:
        """Broadcasts an object to all workers.

        Args:
            obj (Any): object to broadcast to all workers.
            src_rank (int): the rank that broadcasted

        Returns:
            Any: broadcasted object.
        """
        return obj


class RayTorchDistributedStrategy(TorchDistributedStrategy):
    """Base class for all ray distributed strategies."""


class RayDDPStrategy(TorchDDPStrategy, RayTorchDistributedStrategy):
    """A distributed data-parallel (DDP) strategy using Ray Train for PyTorch training."""

    def __init__(self) -> None:
        initialize_ray()
        import ray.train

        self.ray_train = ray.train
        self.name = "ray-torch-ddp"

    def init(self) -> None:
        """Initializes Ray trial/worker.

        Raises:
            RuntimeError: when the Ray cluster is not detected.
        """
        if not ray_cluster_is_running():
            raise RuntimeError("Ray cluster was not detected")
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        self.is_initialized = True

    @check_initialized
    def global_world_size(self) -> int:
        return self.ray_train.get_context().get_world_size()

    @check_initialized
    def local_world_size(self) -> int:
        return self.ray_train.get_context().get_local_world_size()

    @check_initialized
    def global_rank(self) -> int:
        return self.ray_train.get_context().get_world_rank()

    @check_initialized
    def local_rank(self) -> int:
        return self.ray_train.get_context().get_local_rank()

    @check_initialized
    def distributed(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None = None,
    ) -> Tuple[nn.Module, Optimizer, LRScheduler | None]:
        model = self.ray_train.torch.prepare_model(model)

        return model, optimizer, lr_scheduler


class RayDeepSpeedStrategy(DeepSpeedStrategy, RayTorchDistributedStrategy):
    """A distributed strategy using Ray and DeepSpeed for PyTorch training.

    Args:
        backend (Literal["nccl", "gloo", "mpi"]): The backend for distributed communication.
    """

    def __init__(self, backend: Literal["nccl", "gloo", "mpi"]) -> None:
        initialize_ray()
        super().__init__(backend=backend)
        self.name = "ray-deepspeed"

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Raises:
            RuntimeError: when there is not a Ray cluster running.
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        import deepspeed

        self.deepspeed = deepspeed
        if not ray_cluster_is_running():
            raise RuntimeError("Ray cluster was not detected")

        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")

        # https://github.com/Lightning-AI/pytorch-lightning/issues/13567
        # This block of code should be removed as some point
        if os.environ.get("LOCAL_RANK"):
            os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK")

        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        self.deepspeed.init_distributed(dist_backend=self.backend)
        self.is_initialized = True

        self.set_device()

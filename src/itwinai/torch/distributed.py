import abc
from typing import Any, List, Optional, Tuple, Union, Iterable, Literal
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
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
from torch.utils.data.dataloader import T_co, _worker_init_fn_t, _collate_fn_t

from ..distributed import DistributedStrategy
from .type import UninitializedStrategyError, DistributedStrategyError


def distributed_resources_available() -> bool:
    """Check if the current execution environment
    has (enough) GPUs available to allow for distributed ML.

    Returns:
        bool: env can support distributed ML.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return True
    return False


class TorchDistributedStrategy(DistributedStrategy):
    """Abstract class to define the distributed backend methods for
    PyTorch models.
    """

    #: Allows to discriminate distributed strategies from non-distributed.
    #: Defaults to True.
    is_distributed: bool = True
    #: Set to True when the current strategy is initialized.
    #: Defaults to False.
    is_initialized: bool = False

    @property
    def is_main_worker(self) -> bool:
        """Checks if local worker has global rank equal to zero.

        Returns:
            bool: True if main worker.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return self.global_rank() == 0

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

    def device(self) -> str:
        """Device used by local worker.

        Returns:
            str: torch device in the form 'cuda:N'.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return f"cuda:{self.local_rank()}"

    def set_device(self):
        """Set local device."""
        torch.cuda.device(self.local_rank())
        # Needed by torch.distributed.gather_object
        torch.cuda.set_device(self.local_rank())

    def create_dataloader(
        self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False, drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None, generator=None,
        *, prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
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
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")

        if self.is_distributed:
            if sampler is not None:
                raise RuntimeError(
                    "User-provided sampler is not supported."
                )
            sampler = DistributedSampler(
                dataset, num_replicas=self.global_world_size(),
                rank=self.global_rank(),
                shuffle=shuffle
            )
        # shuffle and batch_sampler must be unset
        return DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
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


class TorchDDPStrategy(TorchDistributedStrategy):
    """PyTorch ``DistributedDataParallel`` distributed strategy class.

    Args:
        backend (Literal['nccl', 'gloo', 'mpi']): Name of the
            distributed communication backend to employ.
    """

    #: Torch distributed communication backend.
    backend: Literal['nccl', 'gloo', 'mpi']

    def __init__(self, backend: Literal['nccl', 'gloo', 'mpi']) -> None:
        super().__init__()
        self.backend = backend

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                which is already initialized.
        """
        if not distributed_resources_available():
            raise RuntimeError(
                "Trying to run distributed on insufficient resources.")
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        dist.init_process_group(backend=self.backend)
        self.is_initialized = True

        self.set_device()

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
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if torch.cuda.is_available():
            # device = self.dist_lrank()
            model = model.to(self.device())
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device()],
                output_device=self.device()
            )
        else:
            dist_model = model

        return dist_model, optimizer, lr_scheduler

    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_world_size()

    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return torch.cuda.device_count()

    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_rank()

    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()

    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        res = [None] * self.global_world_size()
        dist.all_gather_object(res, obj)
        return res

    def gather_obj(self, obj: Any, dst_rank: int = 0) -> Optional[List[Any]]:
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
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if self.global_rank() == dst_rank:
            res = [None] * self.global_world_size()
            dist.gather_object(obj, res, dst=dst_rank)
            return res

        dist.gather_object(obj, dst=dst_rank)


class DeepSpeedStrategy(TorchDistributedStrategy):
    """DeepSpeed distributed strategy class.

    Args:
        backend (Literal['nccl', 'gloo', 'mpi']): Name of the
            distributed communication backend to employ.
        config (Union[dict, Path, str]): DeepSpeed config. Either a
            dictionary or a path to a JSON file.
    """

    #: Torch distributed communication backend.
    backend: Literal['nccl', 'gloo', 'mpi']

    def __init__(
        self,
        backend: Literal['nccl', 'gloo', 'mpi']
    ) -> None:
        super().__init__()
        self.backend = backend

    def _load_config(self, ds_config) -> None:
        if isinstance(ds_config, (str, Path)):
            with open(ds_config) as fp:
                self.config = json.load(fp)
        elif isinstance(ds_config, dict):
            self.config = ds_config
        else:
            raise ValueError("ds_config is neither a dictionary not a path.")

    def init(self) -> None:
        """Initializes the distributed process group and the distributed
        package.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        if not distributed_resources_available():
            raise RuntimeError(
                "Trying to run distributed on insufficient resources.")

        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")

        # https://github.com/Lightning-AI/pytorch-lightning/issues/13567
        ompi_lrank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get(
            'LOCAL_RANK', ompi_lrank)

        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        deepspeed.init_distributed(dist_backend=self.backend)
        self.is_initialized = True

        self.set_device()

    def distributed(
        self, model: nn.Module, optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        model_parameters: Optional[Any] = None,
        **init_kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")

        if init_kwargs.get("config"):
            self._load_config(init_kwargs.get("config"))
        # https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization
        # To prioritize optim in the config, you need to pass optim=None
        distrib_model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
            **init_kwargs
        )
        return distrib_model, optimizer, lr_scheduler

    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_world_size()

    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return torch.cuda.device_count()

    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_rank()

    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        # deepspeed.sys.exit() # disabled as it kills the execution

    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gathers any object from the whole group
        in a list (to all workers).

        Args:
            obj (Any): Object to gather from all workers.

        Returns:
            List[Any]: List of gathered objects.
        """
        # https://pytorch.org/docs/stable/distributed.html#collective-functions
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        res = [None] * self.global_world_size()
        dist.all_gather_object(res, obj)
        return res

    def gather_obj(self, obj: Any, dst_rank: int = 0) -> Optional[List[Any]]:
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
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if self.global_rank() == dst_rank:
            res = [None] * self.global_world_size()
            dist.gather_object(obj, res, dst=dst_rank)
            return res

        dist.gather_object(obj, dst=dst_rank)


class HorovodStrategy(TorchDistributedStrategy):
    """Horovod distributed strategy class."""

    def init(self) -> None:
        """Initializes the Horovod distributed backend.

        Raises:
            RuntimeError: when there are not (enough) GPUs available.
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        if not distributed_resources_available():
            raise RuntimeError(
                "Trying to run distributed on insufficient resources.")
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        hvd.init()
        self.is_initialized = True

        self.set_device()

    def distributed(
        self, model: nn.Module, optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        **optim_kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Setup model, optimizer and scheduler for distributed."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")

        model.to(self.device())

        # Scale learning rate
        # https://github.com/horovod/horovod/issues/1653#issuecomment-574764452
        lr_scaler = 1
        if optim_kwargs.get('op') == hvd.Adasum:
            lr_scaler = hvd.local_size()
        elif optim_kwargs.get('op') == hvd.Average:
            lr_scaler = hvd.size()
        for g in optimizer.param_groups:
            g['lr'] *= lr_scaler

        self._broadcast_params(model, optimizer)

        distOptimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            **optim_kwargs
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

    def global_world_size(self) -> int:
        """Returns the total number of processes (global world size).

        Returns:
            int: global world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return hvd.size()

    def local_world_size(self) -> int:
        """Returns the local number of workers available per node,
        which is usually the number of GPUs available.

        Returns:
            int: local world size.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return hvd.local_size()

    def global_rank(self) -> int:
        """Returns the global rank of the current process, where
        rank ranges from 0 to world_size.

        Returns:
            int: global rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return hvd.rank()

    def local_rank(self) -> int:
        """Returns the local rank of the current process.

        Returns:
            int: local rank.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return hvd.local_rank()

    def clean_up(self) -> None:
        """Shuts Horovod down."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        hvd.shutdown()

    def allgather_obj(self, obj: Any) -> list[Any]:
        """All-gathers scalar objects across all workers to a
        list with size(#worker), uses horovod communicator

        Args:
            obj (Any): object in a worker.

        Returns:
            list: gathered list with size(#worker).
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return hvd.allgather_object(obj)

    def gather_obj(self, obj: Any, dst_rank: int = 0) -> list[Any]:
        """The same as ``allgather_obj``, as gather is not supported
        by Horovod.

        Args:
            obj (Any): object in a worker.
            dst_rank (int): ignored.

        Returns:
            list: gathered list with size(#worker).
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        return self.allgather_obj(obj)


class NonDistributedStrategy(TorchDistributedStrategy):
    """Dummy class for non-distributed environments."""

    #: This strategy is not distributed.
    #: Defaults to False.
    is_distributed: bool = True
    is_distributed: bool = False

    def init(self) -> None:
        """If CUDA is available set CUDA device, and do nothing more.

        Raises:
            DistributedStrategyError: when trying to initialize a strategy
                already initialized.
        """
        if self.is_initialized:
            raise DistributedStrategyError("Strategy was already initialized")
        if torch.cuda.is_available():
            self.set_device()
        self.is_initialized = True

    def device(self) -> str:
        """Device used by local worker.

        Returns:
            str: cpu device if CUDA is not available.
        """
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if torch.cuda.is_available():
            return super().device()
        return "cpu"

    def distributed(
        self, model: nn.Module, optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        **kwargs
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        """Do nothing and return model, optimizer and scheduler."""
        if not self.is_initialized:
            raise UninitializedStrategyError(
                "Strategy has not been initialized. Use the init method.")
        if torch.cuda.is_available():
            model = model.cuda()
        return model, optimizer, lr_scheduler

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

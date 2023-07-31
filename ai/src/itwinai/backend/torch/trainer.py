from abc import abstractmethod

from typing import Optional, Dict, Union, Iterable
from enum import Enum, EnumMeta

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from ..components import Trainer
from .utils import seed_worker


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


class TorchTrainer(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Assumes to be executed in a SLURM cluster with torchrun. Use the torch
    elastic version of DDP:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
    """

    def __init__(
            self,
            model: nn.Module,
            epochs: int,
            # learning_rate: float = 1e-3,
            # optim_kwargs: Optional[Dict] = None,
            testrun: bool = False,
            shuffle_data: bool = False,
            seed: Optional[int] = None,
            log_int: int = 10,
            strategy: Optional[TorchDistributedStrategy] = None,
            backend: TorchDistributedBackend = 'nccl',
            use_cuda: bool = True,
            benchrun: bool = False
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.testrun = testrun
        self.seed = seed
        self.shuffle_data = shuffle_data
        self.log_int = log_int
        self.strategy = strategy
        self.backend = backend
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        # self.learning_rate = learning_rate
        # self.optim_kwargs = optim_kwargs

        self.cuda = self.use_cuda and torch.cuda.is_available()

        # Init distributed backend
        if self.strategy is not None:
            dist.init_process_group(backend=self.backend)

        # get job rank info - rank==0 master gpu
        if torch.cuda.is_available():
            # local world size - per node
            self.lwsize = torch.cuda.device_count() if self.cuda else 0
            # global world size - per run
            self.gwsize = dist.get_world_size()
            # global rank - assign per run
            self.grank = dist.get_rank()
            # local rank - assign per node
            self.lrank = dist.get_rank() % self.lwsize
        else:
            self.gwsize = 1
            self.grank = 0
            self.lrank = 0

        # encapsulate the model on the GPU assigned to the current process
        self.device = torch.device(
            'cuda' if self.cuda and torch.cuda.is_available() else 'cpu',
            self.lrank
        )
        if self.cuda:
            torch.cuda.set_device(self.lrank)

        if self.testrun:
            # deterministic testrun
            torch.manual_seed(self.seed)
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            if self.cuda:
                torch.cuda.manual_seed(self.seed)

        self.model = self.model.to(self.device)
        # Create distributed model
        if self.strategy == TorchDistributedStrategy.NONE:
            pass
        elif self.strategy == TorchDistributedStrategy.DDP:
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device
            )
        else:
            raise NotImplementedError("Only DDP strategy is implemented.")

        # # Optimizer
        # self.optimizer = ...

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, backend_name: str) -> None:
        if backend_name not in TorchDistributedBackend:
            raise ValueError(
                "Unrecognized 'backend' field. Allowed values "
                f"are: {TorchDistributedBackend.list()}")
        self._backend = backend_name

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in TorchDistributedStrategy:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {TorchDistributedStrategy.list()}")
        self._strategy = strategy_name

    def setup(self, args) -> None:
        pass

    def execute(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def _preproc_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.

        Args:
            dataloader (DataLoader): some torch DataLoader instance.
        """
        generator = self.g if self.testrun else dataloader.generator

        if self.strategy is None:
            # No distributed strategy used.
            return DataLoader(
                dataloader.dataset,
                shuffle=self.shuffle_data,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device
            )
        else:
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.gwsize,
                rank=self.grank,
                shuffle=self.shuffle_data
            )
        # Recreate dataloader, with updated sampler
        return DataLoader(
            dataloader.dataset,
            shuffle=self.shuffle_data,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )

    # @abstractmethod
    # def configure_optimizers(self) -> Union[Optimizer, Iterable[Optimizer]]:
    #     pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None
    ):
        train_dataloader = self._preproc_dataloader(train_dataloader)
        if validation_dataloader is not None:
            validation_dataloader = self._preproc_dataloader(
                validation_dataloader
            )

        # self._optimizers = self.configure_optimizers()

        self.model.train()
        for _ in range(self.epochs):
            for tr_b_idx, train_batch in enumerate(train_dataloader):
                self.training_step(batch=train_batch, batch_idx=tr_b_idx)
            if validation_dataloader is not None:
                for val_b_idx, val_batch in enumerate(validation_dataloader):
                    self.validation_step(batch=val_batch, batch_idx=val_b_idx)

    # def optimizers(self) -> Union[Optimizer, Iterable[Optimizer]]:
    #     """Get optimizers"""
    #     return self._optimizers

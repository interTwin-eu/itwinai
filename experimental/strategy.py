import os
import abc
from typing import Any, Optional

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group

# from lightning.pytorch.plugins.environments import ClusterEnvironment
from cluster import ClusterEnvironment, detect_cluster


class Strategy(abc.ABC):
    cluster: ClusterEnvironment

    @property
    @abc.abstractmethod
    def device(self) -> int:
        """Device used by this worker"""

    @abc.abstractmethod
    def setup(self) -> None:
        """Setup the strategy once in a distributed environment."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Frees the distributed strategy resources."""

    @abc.abstractmethod
    def is_main_worker(self) -> bool:
        """Returns True if called from the main process of the pool."""

    @abc.abstractmethod
    def _is_env_setup(self) -> bool:
        """Checks whether the distributed environment is correctly setup."""

    @abc.abstractmethod
    def distribute_model(self, model: Any) -> Any:
        """Distributes a neural network."""

    @abc.abstractmethod
    def distribute_optimizer(self, optimizer: Any) -> Any:
        """Distributes an optimizer."""

    @abc.abstractmethod
    def distribute_dataloader(self, dataloader: Any) -> Any:
        """Distributes a dataloader."""


class DDPStrategy(Strategy):
    def __init__(
        self,
        backend: str = 'nccl',
        cluster: Optional[ClusterEnvironment] = None
    ) -> None:
        super().__init__()
        self.cluster = cluster
        self.backend = backend

    @property
    def device(self) -> int:
        """Returns the local rank. Assumes one worker per GPU."""
        return self.cluster.local_rank()

    def setup(self) -> None:
        """Setup the strategy in a distributed context."""
        if not self._is_env_setup():
            raise RuntimeError(
                "Distributed environment not setup correctly. Use a launcher.")

        # detect_cluster() is preferred
        if self.cluster is None:
            self.cluster = detect_cluster()
        print(f"DDPStrategy executed on '{self.cluster}' cluster")

        # Initializes the default distributed process group
        # and the distributed package
        init_process_group(backend=self.backend)

    def teardown(self) -> None:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    def _is_env_setup(self) -> bool:
        if (os.environ.get('RANK') is not None):
            # and torch.distributed.is_available()):
            return True
        return False

    def is_main_worker(self) -> bool:
        return self.cluster.global_rank() == 0

    def distribute_model(self, model: nn.Module) -> nn.Module:
        model = model.to(f"cuda:{self.device}")
        return DistributedDataParallel(
            model,
            device_ids=[self.device],
            output_device=self.device
        )

    def distribute_optimizer(
        self,
        optimizer: optim.Optimizer
    ) -> optim.Optimizer:
        return optimizer

    def distribute_dataloader(
        self,
        dataloader: DataLoader,
        shuffle: bool = True
    ) -> DataLoader:
        """Makes a torch DataLoader distributed by substituting its sampler."""
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.cluster.world_size(),
            rank=self.cluster.global_rank(),
            shuffle=shuffle
        )
        # Recreate dataloader, with updated sampler
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=dataloader.generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )


class LocalStrategy(Strategy):
    ...


class HorovodStrategy(Strategy):
    ...


class DeepSpeedStrategy(Strategy):
    ...

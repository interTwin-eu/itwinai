import os
import abc
import datetime
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch import optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent
)
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    DynamicRendezvousHandler
)
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
    C10dRendezvousBackend
)
from torch.distributed import TCPStore, init_process_group
from torch.distributed.elastic.multiprocessing import Std

from lightning.pytorch.plugins.environments import (
    ClusterEnvironment, SLURMEnvironment,
    TorchElasticEnvironment, LightningEnvironment
)


class LocalEnvironment(LightningEnvironment):
    ...


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
        cluster: ClusterEnvironment,
        backend: str = 'nccl'
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


# ################## USER CODE ################## #


class UniformRndDataset(Dataset):
    def __init__(self, x_size: int, y_size: int, len: int = 100):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.rand(self.x_size), torch.rand(self.y_size)


def trainer_entrypoint_fn(a, strategy: Strategy):
    strategy.setup()
    print(f"{a}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} {os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")

    # Local model
    model = nn.Linear(3, 4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Distributed model
    model: nn.Module = strategy.distribute_model(model)
    optim: torch.optim.Optimizer = strategy.distribute_optimizer(optim)

    # Data
    train_set = UniformRndDataset(x_size=3, y_size=4)
    train_loader = DataLoader(train_set, batch_size=10, num_workers=1)
    # Distributed dataloader
    train_loader: DataLoader = strategy.distribute_dataloader(train_loader)

    for epoch in range(2):
        for (x, y) in train_loader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            optim.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()

            if strategy.is_main_worker():
                print(f"Loss [epoch={epoch}]: {loss.item()}")

    strategy.teardown()
    return 123


STRATEGY = 'ddp'

RUN_ID = "my_run_id"
MIN_NODES = 1
MAX_NODES = 1
NPROC_PRE_NODE = 4
MAX_RESTARTS = 2

if __name__ == "__main__":
    # STRATEGY BUILDER
    # Instantiate ClusterEnv
    if SLURMEnvironment.detect():
        cluster = SLURMEnvironment()
    elif TorchElasticEnvironment.detect():
        cluster = TorchElasticEnvironment()
    elif LocalEnvironment.detect():
        cluster = LocalEnvironment()
    else:
        raise NotImplementedError("Unrecognized cluster env")

    print(cluster)

    # Instantiate Launcher

    # Torch Elastic launcher
    store = TCPStore(host_name="localhost", port=29400,
                     world_size=NPROC_PRE_NODE, is_master=True,
                     timeout=datetime.timedelta(seconds=3))
    backend = C10dRendezvousBackend(store, RUN_ID)
    rdzv_handler = DynamicRendezvousHandler.from_backend(
        run_id=RUN_ID,
        store=store,
        backend=backend,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES
    )

    # Instantiate Strategy
    if STRATEGY == 'ddp' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(cluster=cluster, backend='nccl')
    else:
        raise NotImplementedError

    # CLIENT CODE
    # Launch training from launcher
    spec = WorkerSpec(
        role="trainer",
        local_world_size=NPROC_PRE_NODE,
        entrypoint=trainer_entrypoint_fn,
        args=("foobar", strategy),
        rdzv_handler=rdzv_handler,
        max_restarts=MAX_RESTARTS,
        #   monitor_interval=args.monitor_interval,
        # # redirects={0: Std.ALL} # do no print, but save to file. linked to Agent's log_dir
        redirects=Std.ALL,  # suppress all printing to console
        # tee={0: Std.ALL} reactivates print to console + save to log file for RANK 0
        tee={0: Std.ALL}
    )

    agent = LocalElasticAgent(spec, start_method="spawn", log_dir='logs')
    #   try:
    run_result = agent.run()
    if run_result.is_failed():
        print(f"worker 0 failed with: {run_result.failures[0]}")
    else:
        print(f"worker 0 return value is: {run_result.return_values[0]}")
    #   except Exception ex:
    #       # handle exception

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from strategy import Strategy, DDPStrategy
from launcher import DummyTorchElasticLauncher, TorchElasticLauncher
from launcher_factory import (
    LauncherFactory,
    SimpleLauncherFactory,
    TorchElasticLauncherFactory
)
from distributed_tools import DistributedTooling


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
    """Dummy training function."""
    strategy.setup()
    print(f"{a}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
          f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")

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
            # print(f"tensor to cuda:{strategy.device}")
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


LAUNCHER = 'torch-elastic-no'
STRATEGY = 'ddp'

RUN_ID = "my_run_id"
MIN_NODES = 1
MAX_NODES = 1
NPROC_PRE_NODE = 4
MAX_RESTARTS = 2

if __name__ == "__main__":
    # # STRATEGY BUILDER

    # # Instantiate Launcher Factory
    # # launcher = DummyTorchElasticLauncher(
    # #     n_workers_per_node=NPROC_PRE_NODE,
    # #     min_nodes=MIN_NODES,
    # #     max_nodes=MAX_NODES
    # # )
    # # launcher = TorchElasticLauncher(
    # #     rdzv_id=RUN_ID,
    # #     nproc_per_node=NPROC_PRE_NODE,
    # #     nnodes=f"{MIN_NODES}:{MAX_NODES}",
    # #     max_restarts=MAX_RESTARTS
    # # )
    # if LAUNCHER == 'torch-elastic':
    #     launcher_builder: LauncherFactory = TorchElasticLauncherFactory()
    # else:
    #     launcher_builder: LauncherFactory = SimpleLauncherFactory()

    # # Instantiate launcher
    # launcher = launcher_builder.createLauncher(
    #     n_workers_per_node=NPROC_PRE_NODE
    # )

    # # Instantiate Strategy
    # if (STRATEGY == 'ddp'
    #     and torch.cuda.is_available()
    #         and torch.cuda.device_count() > 1):
    #     strategy = DDPStrategy(cluster=None, backend='nccl')
    # else:
    #     raise NotImplementedError

    dist_tools = DistributedTooling(n_workers_per_node=NPROC_PRE_NODE)
    launcher, strategy = dist_tools.getTools('ddp')

    # CLIENT CODE
    # Launch training from launcher
    launcher.run(func=trainer_entrypoint_fn, args=("foobar", strategy))
